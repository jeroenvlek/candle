#![allow(dead_code)]
// TODO: KV cache.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor, D};
use clap::Parser;
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

mod model;
use model::{Config, Falcon, VarBuilder};

#[cfg(feature = "mkl")]
const DTYPE: DType = DType::F32;
#[cfg(not(feature = "mkl"))]
const DTYPE: DType = DType::BF16;

const TEMPERATURE: Option<f64> = None;

struct TextGeneration {
    model: Falcon,
    rng: rand::rngs::StdRng,
    device: Device,
    tokenizer: Tokenizer,
}

impl TextGeneration {
    fn new(model: Falcon, tokenizer: Tokenizer, seed: u64, device: &Device) -> Self {
        Self {
            model,
            tokenizer,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        println!("starting the inference loop");
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let start_gen = std::time::Instant::now();
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = if let Some(temperature) = TEMPERATURE {
                let prs = (&logits / temperature)?.softmax(D::Minus1)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            new_tokens.push(next_token);
            println!("> {:?}", start_gen.elapsed());
            println!(
                "{} token: {} '{}'",
                index + 1,
                next_token,
                self.tokenizer
                    .decode(vec![next_token], true)
                    .map_err(E::msg)?
            );
        }
        let dt = start_gen.elapsed();
        println!(
            "{sample_len} tokens generated ({} token/s)\n----\n{}\n----",
            sample_len as f64 / dt.as_secs_f64(),
            self.tokenizer.decode(new_tokens, true).map_err(E::msg)?
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    prompt: String,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    #[arg(long, default_value = "tiiuae/falcon-7b")]
    model_id: String,

    #[arg(long, default_value = "refs/pr/43")]
    revision: String,
}

fn main() -> Result<()> {
    use candle_hub::{api::sync::Api, Repo, RepoType};

    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = Repo::with_revision(args.model_id, RepoType::Model, args.revision);
    let tokenizer_filename = api.get(&repo, "tokenizer.json")?;
    let mut filenames = vec![];
    for rfilename in [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        let filename = api.get(&repo, rfilename)?;
        filenames.push(filename);
    }
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let weights = filenames
        .iter()
        .map(|f| Ok(unsafe { candle::safetensors::MmapedFile::new(f)? }))
        .collect::<Result<Vec<_>>>()?;
    let weights = weights
        .iter()
        .map(|f| Ok(f.deserialize()?))
        .collect::<Result<Vec<_>>>()?;

    let vb = VarBuilder::from_safetensors(weights, DTYPE, &device);
    let config = Config::falcon7b();
    config.validate()?;
    let model = Falcon::load(&vb, config)?;
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(model, tokenizer, args.seed, &device);
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}