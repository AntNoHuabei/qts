"""Export Qwen3-TTS main weights to GGUF and the 12Hz vocoder to ONNX."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import onnx
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

import gguf
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

from qwen3_tts_native_scripts.export_voice_clone_prompt import resolve_dtype

logger = logging.getLogger(__name__)

MODEL_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "speech_tokenizer/*.json",
    "speech_tokenizer/*.safetensors",
]

MAIN_TYPE_TO_QUANT = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
}

MAIN_TYPE_TO_FILE_TYPE = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
    "q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
}


class Qwen3MainGgufExporter:
    """Export the talker, code predictor, speaker encoder, and tokenizer metadata to GGUF."""

    TENSOR_MAP = {
        "talker.model.codec_embedding.weight": "talker.codec_embd.weight",
        "talker.model.text_embedding.weight": "talker.text_embd.weight",
        "talker.codec_head.weight": "talker.codec_head.weight",
        "talker.model.norm.weight": "talker.output_norm.weight",
        "talker.text_projection.linear_fc1.weight": "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias": "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight": "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias": "talker.text_proj.fc2.bias",
        "talker.code_predictor.model.norm.weight": "code_pred.output_norm.weight",
        "speaker_encoder.blocks.0.conv.weight": "spk_enc.conv0.weight",
        "speaker_encoder.blocks.0.conv.bias": "spk_enc.conv0.bias",
        "speaker_encoder.asp.conv.weight": "spk_enc.asp.conv.weight",
        "speaker_encoder.asp.conv.bias": "spk_enc.asp.conv.bias",
        "speaker_encoder.asp.tdnn.conv.weight": "spk_enc.asp.tdnn.weight",
        "speaker_encoder.asp.tdnn.conv.bias": "spk_enc.asp.tdnn.bias",
        "speaker_encoder.mfa.conv.weight": "spk_enc.mfa.weight",
        "speaker_encoder.mfa.conv.bias": "spk_enc.mfa.bias",
        "speaker_encoder.fc.weight": "spk_enc.fc.weight",
        "speaker_encoder.fc.bias": "spk_enc.fc.bias",
    }

    TALKER_LAYER_PATTERNS = [
        (r"talker\.model\.layers\.(\d+)\.input_layernorm\.weight", "talker.blk.{}.attn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "talker.blk.{}.attn_q.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "talker.blk.{}.attn_k.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "talker.blk.{}.attn_v.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "talker.blk.{}.attn_output.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "talker.blk.{}.attn_q_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "talker.blk.{}.attn_k_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "talker.blk.{}.ffn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "talker.blk.{}.ffn_gate.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "talker.blk.{}.ffn_up.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "talker.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_LAYER_PATTERNS = [
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.input_layernorm\.weight", "code_pred.blk.{}.attn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "code_pred.blk.{}.attn_q.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "code_pred.blk.{}.attn_k.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "code_pred.blk.{}.attn_v.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "code_pred.blk.{}.attn_output.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "code_pred.blk.{}.attn_q_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "code_pred.blk.{}.attn_k_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "code_pred.blk.{}.ffn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "code_pred.blk.{}.ffn_gate.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "code_pred.blk.{}.ffn_up.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "code_pred.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_CODEBOOK_PATTERNS = [
        (r"talker\.code_predictor\.model\.codec_embedding\.(\d+)\.weight", "code_pred.codec_embd.{}.weight"),
        (r"talker\.code_predictor\.lm_head\.(\d+)\.weight", "code_pred.lm_head.{}.weight"),
    ]

    SPEAKER_ENCODER_PATTERNS = [
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.weight", "spk_enc.blk.{}.res2net.{}.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.bias", "spk_enc.blk.{}.res2net.{}.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.weight", "spk_enc.blk.{}.se.conv1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.bias", "spk_enc.blk.{}.se.conv1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.weight", "spk_enc.blk.{}.se.conv2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.bias", "spk_enc.blk.{}.se.conv2.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.weight", "spk_enc.blk.{}.tdnn1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.bias", "spk_enc.blk.{}.tdnn1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.weight", "spk_enc.blk.{}.tdnn2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.bias", "spk_enc.blk.{}.tdnn2.bias"),
    ]

    def __init__(self, input_dir: Path, output_path: Path, output_type: str):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type
        self.config = self._load_config()
        self._extract_params()

    def _load_config(self) -> dict[str, Any]:
        config_path = self.input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _extract_params(self) -> None:
        talker_config = self.config.get("talker_config", {})
        code_predictor_config = talker_config.get("code_predictor_config", {})
        speaker_encoder_config = self.config.get("speaker_encoder_config", {})

        self.hidden_size = talker_config.get("hidden_size", 1024)
        self.intermediate_size = talker_config.get("intermediate_size", 3072)
        self.num_hidden_layers = talker_config.get("num_hidden_layers", 28)
        self.num_attention_heads = talker_config.get("num_attention_heads", 16)
        self.num_kv_heads = talker_config.get("num_key_value_heads", 8)
        self.head_dim = talker_config.get("head_dim", 128)
        self.vocab_size = talker_config.get("vocab_size", 3072)
        self.text_vocab_size = talker_config.get("text_vocab_size", 151936)
        self.text_hidden_size = talker_config.get("text_hidden_size", 2048)
        self.num_code_groups = talker_config.get("num_code_groups", 16)
        self.rms_norm_eps = talker_config.get("rms_norm_eps", 1e-6)
        self.rope_theta = talker_config.get("rope_theta", 1_000_000)
        self.mrope_section = talker_config.get("rope_scaling", {}).get("mrope_section", [24, 20, 20])
        self.code_predictor_num_layers = code_predictor_config.get("num_hidden_layers", 5)
        self.code_predictor_vocab_size = code_predictor_config.get("vocab_size", 2048)
        self.speaker_enc_dim = speaker_encoder_config.get("enc_dim", 1024)
        self.speaker_sample_rate = speaker_encoder_config.get("sample_rate", 24000)
        self.codec_pad_id = talker_config.get("codec_pad_id", 2148)
        self.codec_bos_id = talker_config.get("codec_bos_id", 2149)
        self.codec_eos_id = talker_config.get("codec_eos_token_id", 2150)
        self.model_name = "Qwen3-TTS-12Hz-0.6B"

    def _map_tensor_name(self, hf_name: str) -> str | None:
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]
        for pattern, template in self.TALKER_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                return template.format(match.group(1))
        for pattern, template in self.CODE_PREDICTOR_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                return template.format(match.group(1))
        for pattern, template in self.CODE_PREDICTOR_CODEBOOK_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                return template.format(match.group(1))
        for pattern, template in self.SPEAKER_ENCODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                return template.format(*groups)
        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        safetensor_files = sorted(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir}")
        for sf_path in safetensor_files:
            logger.info("Loading tensors from %s", sf_path.name)
            with safe_open(sf_path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    yield name, handle.get_tensor(name)

    def _should_quantize(self, tensor_name: str) -> bool:
        if any(x in tensor_name for x in ["_embd", "codebook"]):
            return False
        if "_norm" in tensor_name:
            return False
        if ".bias" in tensor_name:
            return False
        if "lm_head" in tensor_name or "codec_head" in tensor_name:
            return False
        return True

    def _convert_dtype(
        self,
        tensor: torch.Tensor,
        tensor_name: str,
    ) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        data = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        if data.ndim <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        if self.output_type in ("f32", "f16"):
            quant = MAIN_TYPE_TO_QUANT[self.output_type]
            dtype = np.float32 if self.output_type == "f32" else np.float16
            return data.astype(dtype), quant
        if not self._should_quantize(tensor_name):
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        quant = MAIN_TYPE_TO_QUANT[self.output_type]
        try:
            quantized = gguf.quants.quantize(data.astype(np.float32), quant)
            return quantized, quant
        except Exception as exc:
            logger.warning(
                "Quantization failed for %s with %s: %s; falling back to F16",
                tensor_name,
                self.output_type,
                exc,
            )
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    def _load_tokenizer(self) -> tuple[list[str], list[int], list[str]]:
        vocab_path = self.input_dir / "vocab.json"
        merges_path = self.input_dir / "merges.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        vocab_dict = json.loads(vocab_path.read_text(encoding="utf-8"))
        sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])

        tokens: list[str] = []
        toktypes: list[int] = []
        for token, _token_id in sorted_vocab:
            tokens.append(token)
            if token.startswith("<|") and token.endswith("|>"):
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.NORMAL)

        while len(tokens) < self.text_vocab_size:
            tokens.append(f"[PAD{len(tokens)}]")
            toktypes.append(gguf.TokenType.UNUSED)

        merges: list[str] = []
        if merges_path.exists():
            for line in merges_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    merges.append(line)
        return tokens, toktypes, merges

    def _add_metadata(self, writer: gguf.GGUFWriter) -> None:
        arch = "qwen3-tts"
        writer.add_name(self.model_name)
        writer.add_type(gguf.GGUFType.MODEL)
        writer.add_file_type(MAIN_TYPE_TO_FILE_TYPE[self.output_type])
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        writer.add_block_count(self.num_hidden_layers)
        writer.add_embedding_length(self.hidden_size)
        writer.add_feed_forward_length(self.intermediate_size)
        writer.add_head_count(self.num_attention_heads)
        writer.add_head_count_kv(self.num_kv_heads)
        writer.add_key_length(self.head_dim)
        writer.add_value_length(self.head_dim)
        writer.add_rope_freq_base(self.rope_theta)
        writer.add_layer_norm_rms_eps(self.rms_norm_eps)
        writer.add_vocab_size(self.vocab_size)

        writer.add_uint32(f"{arch}.text_vocab_size", self.text_vocab_size)
        writer.add_uint32(f"{arch}.text_hidden_size", self.text_hidden_size)
        writer.add_uint32(f"{arch}.num_code_groups", self.num_code_groups)
        writer.add_array(f"{arch}.rope.mrope_section", self.mrope_section)
        writer.add_uint32(f"{arch}.code_predictor.layer_count", self.code_predictor_num_layers)
        writer.add_uint32(f"{arch}.code_predictor.vocab_size", self.code_predictor_vocab_size)
        writer.add_uint32(f"{arch}.speaker_encoder.embedding_length", self.speaker_enc_dim)
        writer.add_uint32(f"{arch}.speaker_encoder.sample_rate", self.speaker_sample_rate)
        writer.add_uint32(f"{arch}.codec.pad_id", self.codec_pad_id)
        writer.add_uint32(f"{arch}.codec.bos_id", self.codec_bos_id)
        writer.add_uint32(f"{arch}.codec.eos_id", self.codec_eos_id)

    def _add_tokenizer(self, writer: gguf.GGUFWriter) -> None:
        tokens, toktypes, merges = self._load_tokenizer()
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)
        if merges:
            writer.add_token_merges(merges)

        tokenizer_config_path = self.input_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            vocab = json.loads((self.input_dir / "vocab.json").read_text(encoding="utf-8"))

            eos_token = tokenizer_config.get("eos_token")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")
            if eos_token in vocab:
                writer.add_eos_token_id(vocab[eos_token])

            pad_token = tokenizer_config.get("pad_token")
            if isinstance(pad_token, dict):
                pad_token = pad_token.get("content")
            if pad_token in vocab:
                writer.add_pad_token_id(vocab[pad_token])

            chat_template = tokenizer_config.get("chat_template")
            if chat_template:
                writer.add_chat_template(chat_template)

    def export(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(path=None, arch="qwen3-tts")
        self._add_metadata(writer)
        self._add_tokenizer(writer)

        tensor_count = 0
        skipped_count = 0
        for hf_name, tensor in tqdm(self._get_tensors(), desc="Converting GGUF"):
            ggml_name = self._map_tensor_name(hf_name)
            if ggml_name is None:
                skipped_count += 1
                logger.debug("Skipping unmapped tensor: %s", hf_name)
                continue
            data, dtype = self._convert_dtype(tensor, ggml_name)
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

        logger.info(
            "Prepared GGUF tensors: converted=%s skipped=%s output=%s",
            tensor_count,
            skipped_count,
            self.output_path,
        )
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        return self.output_path


class VocoderOnnxWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, decode_upsample_rate: int):
        super().__init__()
        self.decoder = decoder
        self.decode_upsample_rate = int(decode_upsample_rate)

    def forward(self, audio_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_lengths = (audio_codes[..., 0] > -1).sum(dim=1) * self.decode_upsample_rate
        clamped = torch.clamp(audio_codes, min=0).to(dtype=torch.long)
        audio_values = self.decoder(clamped.transpose(1, 2)).squeeze(1)
        return audio_values, audio_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument("--out-dir", required=True, help="Directory to write exported artifacts into.")
    parser.add_argument(
        "--main-type",
        default="f16",
        choices=sorted(MAIN_TYPE_TO_QUANT),
        help="Quantization/data type for the main GGUF export.",
    )
    parser.add_argument("--main-out", default=None, help="Override the main GGUF output path.")
    parser.add_argument("--vocoder-out", default=None, help="Override the vocoder ONNX output path.")
    parser.add_argument(
        "--vocoder-dtype",
        default="float32",
        help="dtype used when loading the PyTorch vocoder before ONNX export.",
    )
    parser.add_argument(
        "--vocoder-opset",
        type=int,
        default=17,
        help="ONNX opset version for vocoder export.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download from Hugging Face; require all files to already exist locally.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def resolve_model_dir(model_name_or_path: str, local_files_only: bool) -> Path:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return path.resolve()
    snapshot = snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=MODEL_ALLOW_PATTERNS,
        local_files_only=local_files_only,
    )
    return Path(snapshot)


def default_main_output(out_dir: Path, main_type: str) -> Path:
    return out_dir / f"qwen3-tts-0.6b-{main_type}.gguf"


def default_vocoder_output(out_dir: Path) -> Path:
    return out_dir / "qwen3-tts-vocoder.onnx"


def add_onnx_metadata(
    onnx_path: Path,
    *,
    source_model: str,
    speech_tokenizer_dir: Path,
    num_quantizers: int,
    decode_upsample_rate: int,
    output_sample_rate: int,
) -> None:
    model = onnx.load(str(onnx_path))
    metadata = {
        "source_model": source_model,
        "speech_tokenizer_dir": str(speech_tokenizer_dir),
        "num_quantizers": str(num_quantizers),
        "decode_upsample_rate": str(decode_upsample_rate),
        "output_sample_rate_hz": str(output_sample_rate),
        "input_layout": "batch,frames,quantizers",
        "output_layout": "batch,samples",
    }
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, str(onnx_path))


def export_vocoder_onnx(
    *,
    speech_tokenizer_dir: Path,
    output_path: Path,
    source_model: str,
    dtype_name: str,
    opset: int,
) -> Path:
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(speech_tokenizer_dir),
        device_map="cpu",
        dtype=resolve_dtype(dtype_name),
        attn_implementation="eager",
    )
    model = tokenizer.model
    model.eval()
    if model.get_model_type() != "qwen3_tts_tokenizer_12hz":
        raise SystemExit(
            f"Only the 12Hz speech tokenizer is supported for ONNX export, got {model.get_model_type()}"
        )

    num_quantizers = int(model.config.decoder_config.num_quantizers)
    decode_upsample_rate = int(model.get_decode_upsample_rate())
    output_sample_rate = int(model.get_output_sample_rate())
    wrapper = VocoderOnnxWrapper(model.decoder, decode_upsample_rate).cpu().eval()
    dummy_codes = torch.zeros((1, 16, num_quantizers), dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_codes,),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["audio_codes"],
        output_names=["audio_values", "audio_lengths"],
        dynamic_axes={
            "audio_codes": {0: "batch", 1: "frames"},
            "audio_values": {0: "batch", 1: "samples"},
            "audio_lengths": {0: "batch"},
        },
    )
    add_onnx_metadata(
        output_path,
        source_model=source_model,
        speech_tokenizer_dir=speech_tokenizer_dir,
        num_quantizers=num_quantizers,
        decode_upsample_rate=decode_upsample_rate,
        output_sample_rate=output_sample_rate,
    )
    logger.info(
        "Exported vocoder ONNX: path=%s num_quantizers=%s decode_upsample_rate=%s",
        output_path,
        num_quantizers,
        decode_upsample_rate,
    )
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    model_dir = resolve_model_dir(args.model, local_files_only=args.local_files_only)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    main_out = Path(args.main_out).expanduser().resolve() if args.main_out else default_main_output(out_dir, args.main_type)
    vocoder_out = Path(args.vocoder_out).expanduser().resolve() if args.vocoder_out else default_vocoder_output(out_dir)

    logger.info("Resolved model dir: %s", model_dir)
    logger.info("Main GGUF output: %s", main_out)
    logger.info("Vocoder ONNX output: %s", vocoder_out)

    if main_out.exists():
        logger.info("Reusing existing main GGUF: %s", main_out)
        gguf_out = main_out
    else:
        gguf_out = Qwen3MainGgufExporter(
            input_dir=model_dir,
            output_path=main_out,
            output_type=args.main_type,
        ).export()

    if vocoder_out.exists():
        logger.info("Reusing existing vocoder ONNX: %s", vocoder_out)
        onnx_out = vocoder_out
    else:
        onnx_out = export_vocoder_onnx(
            speech_tokenizer_dir=model_dir / "speech_tokenizer",
            output_path=vocoder_out,
            source_model=args.model,
            dtype_name=args.vocoder_dtype,
            opset=args.vocoder_opset,
        )

    print(
        f"exported artifacts: main_gguf={gguf_out} "
        f"vocoder_onnx={onnx_out} main_type={args.main_type}"
    )


if __name__ == "__main__":
    main()
