//! Rolling KV cache with sliding-window compaction.

#[derive(Debug, Clone)]
struct LayerKv {
    keys: Vec<f32>,
    values: Vec<f32>,
    len_tokens: usize,
    pos_offset: usize,
}

impl LayerKv {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            len_tokens: 0,
            pos_offset: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingKvCache {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    layers: Vec<LayerKv>,
}

impl RollingKvCache {
    #[must_use]
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, sliding_window: usize) -> Self {
        debug_assert!(n_layers > 0);
        debug_assert!(n_kv_heads > 0);
        debug_assert!(head_dim > 0);
        debug_assert!(sliding_window > 0);

        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            sliding_window,
            layers: vec![LayerKv::new(); n_layers],
        }
    }

    pub fn append_layer(
        &mut self,
        layer: usize,
        k_new: &[f32],
        v_new: &[f32],
        n_new_tokens: usize,
    ) {
        debug_assert!(layer < self.n_layers);
        if n_new_tokens == 0 {
            return;
        }
        let stride = self.n_kv_heads * self.head_dim;
        debug_assert_eq!(k_new.len(), n_new_tokens * stride);
        debug_assert_eq!(v_new.len(), n_new_tokens * stride);

        let cache = &mut self.layers[layer];
        let requested = cache.len_tokens + n_new_tokens;
        let drop_tokens = requested.saturating_sub(self.sliding_window);

        if drop_tokens > 0 {
            let keep_tokens = cache.len_tokens.saturating_sub(drop_tokens);
            let keep_elems = keep_tokens * stride;
            if keep_tokens == 0 {
                cache.keys.clear();
                cache.values.clear();
                cache.len_tokens = 0;
            } else {
                let drop_elems = drop_tokens * stride;
                cache.keys.drain(0..drop_elems);
                cache.values.drain(0..drop_elems);
                cache.keys.truncate(keep_elems);
                cache.values.truncate(keep_elems);
                cache.len_tokens = keep_tokens;
            }
            cache.pos_offset += drop_tokens;
        }

        cache.keys.extend_from_slice(k_new);
        cache.values.extend_from_slice(v_new);
        cache.len_tokens += n_new_tokens;
    }

    pub fn layer_len_tokens(&self, layer: usize) -> usize {
        self.layers[layer].len_tokens
    }

    pub fn layer_pos_offset(&self, layer: usize) -> usize {
        self.layers[layer].pos_offset
    }

    pub fn layer_tensors(&self, layer: usize) -> (&[f32], &[f32]) {
        let cache = &self.layers[layer];
        (&cache.keys, &cache.values)
    }
}

#[cfg(test)]
mod tests {
    use super::RollingKvCache;

    #[test]
    fn compacts_when_exceeding_window() {
        let mut kv = RollingKvCache::new(1, 2, 2, 4); // stride=4

        let k1: Vec<f32> = (0..(3 * 4)).map(|x| x as f32).collect();
        let v1: Vec<f32> = (100..(100 + 3 * 4)).map(|x| x as f32).collect();
        kv.append_layer(0, &k1, &v1, 3);
        assert_eq!(kv.layer_len_tokens(0), 3);
        assert_eq!(kv.layer_pos_offset(0), 0);

        let k2: Vec<f32> = (1000..(1000 + 3 * 4)).map(|x| x as f32).collect();
        let v2: Vec<f32> = (2000..(2000 + 3 * 4)).map(|x| x as f32).collect();
        kv.append_layer(0, &k2, &v2, 3);

        // 3+3 with window=4 => drop 2 oldest.
        assert_eq!(kv.layer_len_tokens(0), 4);
        assert_eq!(kv.layer_pos_offset(0), 2);

        let (k, v) = kv.layer_tensors(0);
        assert_eq!(k.len(), 4 * 4);
        assert_eq!(v.len(), 4 * 4);

        // First remaining token should correspond to original absolute token index 2.
        assert_eq!(k[0], k1[2 * 4]);
        assert_eq!(v[0], v1[2 * 4]);
        // Last token should be from second append.
        assert_eq!(k[(4 - 1) * 4], k2[(3 - 1) * 4]);
        assert_eq!(v[(4 - 1) * 4], v2[(3 - 1) * 4]);
    }

    #[test]
    fn layers_are_isolated() {
        let mut kv = RollingKvCache::new(2, 1, 2, 3); // stride=2
        kv.append_layer(0, &[1.0, 2.0], &[3.0, 4.0], 1);
        kv.append_layer(1, &[10.0, 20.0, 30.0, 40.0], &[50.0, 60.0, 70.0, 80.0], 2);

        assert_eq!(kv.layer_len_tokens(0), 1);
        assert_eq!(kv.layer_len_tokens(1), 2);

        let (k0, _) = kv.layer_tensors(0);
        let (k1, _) = kv.layer_tensors(1);
        assert_eq!(k0, &[1.0, 2.0]);
        assert_eq!(k1, &[10.0, 20.0, 30.0, 40.0]);
    }
}
