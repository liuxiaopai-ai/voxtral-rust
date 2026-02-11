use voxtral::model::{ModelBundle, ModelMetadata};

/// Optional integration test.
///
/// Run locally with:
/// `VOXTRAL_MODEL_DIR=/path/to/model cargo test -p voxtral --test e2e_model_env -- --nocapture`
#[test]
fn model_dir_env_smoke() {
    let Some(model_dir) = std::env::var_os("VOXTRAL_MODEL_DIR") else {
        eprintln!("skipping: VOXTRAL_MODEL_DIR is not set");
        return;
    };

    let meta = ModelMetadata::load_from_dir(&model_dir).expect("load metadata");
    assert_eq!(meta.params.vocab_size, 131_072);
    assert_eq!(meta.tokenizer.bos_id(), Some(1));
    assert_eq!(meta.tokenizer.eos_id(), Some(2));
    assert_eq!(meta.tokenizer.streaming_pad_id(), Some(32));

    let bundle = ModelBundle::load_from_dir(&model_dir).expect("load bundle");
    let names = bundle.weights.names().expect("list tensor names");

    assert!(
        names
            .iter()
            .any(|n| n == "mm_streams_embeddings.embedding_module.tok_embeddings.weight"),
        "missing token embedding tensor"
    );
    assert!(
        names.iter().any(|n| n == "layers.0.attention.wq.weight"),
        "missing decoder layer-0 attention tensor"
    );
}
