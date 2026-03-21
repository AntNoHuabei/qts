#include "tts_transformer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

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

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <hidden.bin> <cb0_token>\n";
        return 1;
    }

    qwen3_tts::TTSTransformer transformer;
    if (!transformer.load_model(argv[1])) {
        std::cerr << "load_model failed: " << transformer.get_error() << "\n";
        return 1;
    }

    std::vector<float> hidden = read_f32_file(argv[2]);
    const int32_t cb0 = std::stoi(argv[3]);
    std::vector<int32_t> codes;
    if (!transformer.predict_codes_autoregressive(hidden.data(), cb0, codes, 0.0f, 0)) {
        std::cerr << "predict_codes_autoregressive failed: " << transformer.get_error() << "\n";
        return 1;
    }

    for (size_t i = 0; i < codes.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << codes[i];
    }
    std::cout << '\n';
    return 0;
}
