//! Safetensors-backed model weight loading.

use std::path::Path;

use memmap2::MmapOptions;
use safetensors::tensor::{Dtype, SafeTensorError};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct TensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub struct WeightStore {
    mmap: memmap2::Mmap,
}

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] SafeTensorError),
    #[error("unsupported dtype for {name}: {dtype:?}")]
    UnsupportedDtype { name: String, dtype: Dtype },
    #[error("invalid tensor byte length for {name}: got {bytes}, expected multiple of {elem_size}")]
    InvalidByteLen {
        name: String,
        bytes: usize,
        elem_size: usize,
    },
}

impl WeightStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WeightError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only file mapping for immutable tensor access.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    pub fn names(&self) -> Result<Vec<String>, WeightError> {
        let st = safetensors::SafeTensors::deserialize(&self.mmap)?;
        Ok(st.iter().map(|(name, _)| name.to_string()).collect())
    }

    pub fn tensor_f32(&self, name: &str) -> Result<TensorF32, WeightError> {
        let st = safetensors::SafeTensors::deserialize(&self.mmap)?;
        let tv = st.tensor(name)?;
        let dtype = tv.dtype();
        let shape = tv.shape().to_vec();
        let raw = tv.data();

        let data = match dtype {
            Dtype::F32 => {
                if raw.len() % 4 != 0 {
                    return Err(WeightError::InvalidByteLen {
                        name: name.to_string(),
                        bytes: raw.len(),
                        elem_size: 4,
                    });
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            Dtype::BF16 => {
                if raw.len() % 2 != 0 {
                    return Err(WeightError::InvalidByteLen {
                        name: name.to_string(),
                        bytes: raw.len(),
                        elem_size: 2,
                    });
                }
                raw.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
                        f32::from_bits(bits << 16)
                    })
                    .collect()
            }
            other => {
                return Err(WeightError::UnsupportedDtype {
                    name: name.to_string(),
                    dtype: other,
                });
            }
        };

        Ok(TensorF32 { shape, data })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use safetensors::tensor::{Dtype, View, serialize_to_file};

    use super::WeightStore;

    #[derive(Debug, Clone)]
    struct TestTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for TestTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn tmp_file(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        p.push(format!("voxtral-weights-test-{name}-{nanos}.safetensors"));
        p
    }

    #[test]
    fn loads_f32_and_bf16_tensors() {
        let f32_data = vec![1.0f32, 2.5, -3.0, 4.25];
        let mut f32_bytes = Vec::with_capacity(f32_data.len() * 4);
        for v in &f32_data {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }

        // BF16 for [1.0, -2.0] -> [0x3f80, 0xc000]
        let bf16_words = [0x3f80u16, 0xc000u16];
        let mut bf16_bytes = Vec::with_capacity(bf16_words.len() * 2);
        for &w in &bf16_words {
            bf16_bytes.extend_from_slice(&w.to_le_bytes());
        }

        let tensors = vec![
            (
                "a".to_string(),
                TestTensor {
                    dtype: Dtype::F32,
                    shape: vec![2, 2],
                    data: f32_bytes,
                },
            ),
            (
                "b".to_string(),
                TestTensor {
                    dtype: Dtype::BF16,
                    shape: vec![2],
                    data: bf16_bytes,
                },
            ),
        ];

        let path = tmp_file("basic");
        serialize_to_file(tensors, &None, &path).expect("serialize safetensors");

        let ws = WeightStore::open(&path).expect("open");
        let names = ws.names().expect("names");
        assert!(names.iter().any(|n| n == "a"));
        assert!(names.iter().any(|n| n == "b"));

        let a = ws.tensor_f32("a").expect("tensor a");
        assert_eq!(a.shape, vec![2, 2]);
        assert_eq!(a.data, f32_data);

        let b = ws.tensor_f32("b").expect("tensor b");
        assert_eq!(b.shape, vec![2]);
        assert_eq!(b.data.len(), 2);
        assert!((b.data[0] - 1.0).abs() < 1e-6);
        assert!((b.data[1] + 2.0).abs() < 1e-6);

        std::fs::remove_file(path).expect("cleanup");
    }
}
