#include "tts_transformer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

static std::vector<int32_t> read_i64_tokens(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("failed to open ") + path);
    }
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<int64_t> raw(size / sizeof(int64_t));
    in.read(reinterpret_cast<char *>(raw.data()), size);
    return std::vector<int32_t>(raw.begin(), raw.end());
}

static std::vector<float> read_f32_file(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("failed to open ") + path);
    }
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    in.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

static void write_f32_file(const char * path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <tokens.bin> <speaker.bin> <hidden_out.bin>\n";
        return 1;
    }

    qwen3_tts::TTSTransformer transformer;
    if (!transformer.load_model(argv[1])) {
        std::cerr << "load_model failed: " << transformer.get_error() << "\n";
        return 1;
    }

    auto tokens = read_i64_tokens(argv[2]);
    auto speaker = read_f32_file(argv[3]);
    std::vector<int32_t> codes;
    if (!transformer.generate(tokens.data(), static_cast<int32_t>(tokens.size()),
                              speaker.data(), 1, codes, 2050, 1.05f, 0.0f, 0)) {
        std::cerr << "generate failed: " << transformer.get_error() << "\n";
        return 1;
    }

    std::vector<float> hidden;
    if (!transformer.get_hidden_states(hidden)) {
        std::cerr << "get_hidden_states failed: " << transformer.get_error() << "\n";
        return 1;
    }
    write_f32_file(argv[4], hidden);

    std::cout << "codes:";
    for (int32_t code : codes) {
        std::cout << ' ' << code;
    }
    std::cout << "\nhidden_len: " << hidden.size() << '\n';
    return 0;
}
