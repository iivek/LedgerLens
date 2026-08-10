// LedgerLens JNI bridge around llama.cpp + mtmd (multimodal) for Qwen2-VL GGUF models.
//
// IMPORTANT: llama.cpp's public API — and the mtmd API in particular — moves fast.
// This file is verified against llama.cpp tag b10344 (2026-08-10) -- every symbol
// used below was checked directly against that tag's headers, not assumed; three
// real breaking changes were found and fixed versus an earlier draft written
// against a mid-2025 API: mtmd_context_params lost its `verbosity` field (use
// mtmd_log_set() instead), mtmd_helper_bitmap_init_from_file() now returns a
// mtmd_helper_bitmap_wrapper struct by value (with an extra `placeholder` bool
// param) instead of a raw mtmd_bitmap*, and mtmd_input_text.text_len must be
// set explicitly (a real bug even before this update: it was left uninitialized
// and mtmd.cpp reads it directly). If the build breaks here after bumping the
// submodule to a newer tag, diff against:
//   third_party/llama.cpp/tools/mtmd/mtmd.h
//   third_party/llama.cpp/tools/mtmd/mtmd-helper.h
//   third_party/llama.cpp/tools/mtmd/mtmd-cli.cpp   (reference usage)
//   third_party/llama.cpp/include/llama.h

#include <jni.h>
#include <android/log.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define LOG_TAG "LedgerLensJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Session {
    llama_model *       model  = nullptr;
    llama_context *     lctx   = nullptr;
    mtmd_context *      mctx   = nullptr;
    const llama_vocab * vocab  = nullptr;
    int32_t             n_batch = 0;
    // Belt-and-braces: the Kotlin repository already serializes calls with a Mutex,
    // but the native side must never be re-entered concurrently either.
    std::mutex          mu;
};

std::string to_std_string(JNIEnv * env, jstring js) {
    if (js == nullptr) return {};
    const char * chars = env->GetStringUTFChars(js, nullptr);
    std::string out = chars ? chars : "";
    if (chars) env->ReleaseStringUTFChars(js, chars);
    return out;
}

void throw_runtime_exception(JNIEnv * env, const std::string & msg) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls != nullptr) {
        env->ThrowNew(cls, msg.c_str());
    }
}

void android_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    int prio = ANDROID_LOG_INFO;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG; break;
        default: break;
    }
    __android_log_write(prio, "llama.cpp", text);
}

void ensure_backend_init() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(android_log_callback, nullptr);
        // mtmd has its own log sink, separate from llama's -- without this,
        // mtmd/vision-tower logs (image preprocessing, tokenization errors)
        // never reach logcat at all.
        mtmd_log_set(android_log_callback, nullptr);
        llama_backend_init();
    });
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_ledgerlens_app_LlamaBridge_loadModel(
        JNIEnv * env, jobject /*thiz*/,
        jstring jModelPath, jstring jMmprojPath, jint nCtx, jint nThreads) {

    ensure_backend_init();

    const std::string model_path  = to_std_string(env, jModelPath);
    const std::string mmproj_path = to_std_string(env, jMmprojPath);

    LOGI("loadModel: model=%s mmproj=%s n_ctx=%d n_threads=%d",
         model_path.c_str(), mmproj_path.c_str(), (int) nCtx, (int) nThreads);

    auto session = std::make_unique<Session>();

    // --- text model ---
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU-only

    session->model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (session->model == nullptr) {
        throw_runtime_exception(env, "Failed to load model: " + model_path);
        return 0;
    }
    session->vocab = llama_model_get_vocab(session->model);

    // --- context ---
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) nCtx;
    // Image embeddings are evaluated in batches; keep n_batch generous so a whole
    // image chunk fits in few passes. mtmd_helper_eval_chunks() splits as needed.
    cparams.n_batch         = 2048;
    cparams.n_ubatch        = 2048;
    cparams.n_threads       = (int32_t) nThreads;
    cparams.n_threads_batch = (int32_t) nThreads;

    session->lctx = llama_init_from_model(session->model, cparams);
    if (session->lctx == nullptr) {
        llama_model_free(session->model);
        throw_runtime_exception(env, "Failed to create llama_context");
        return 0;
    }
    session->n_batch = (int32_t) cparams.n_batch;

    // --- multimodal projector (vision tower for Qwen2-VL) ---
    // NOTE: mtmd_context_params no longer has a `verbosity` field (removed
    // upstream) -- log level is controlled via mtmd_log_set() instead, wired
    // up once in ensure_backend_init().
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu       = false;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads     = (int32_t) nThreads;

    session->mctx = mtmd_init_from_file(mmproj_path.c_str(), session->model, mtmd_params);
    if (session->mctx == nullptr) {
        llama_free(session->lctx);
        llama_model_free(session->model);
        throw_runtime_exception(env, "Failed to load mmproj: " + mmproj_path);
        return 0;
    }

    LOGI("loadModel: success");
    return reinterpret_cast<jlong>(session.release());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ledgerlens_app_LlamaBridge_extract(
        JNIEnv * env, jobject /*thiz*/,
        jlong handle, jstring jImagePath, jstring jPrompt, jint maxTokens) {

    auto * session = reinterpret_cast<Session *>(handle);
    if (session == nullptr) {
        throw_runtime_exception(env, "extract() called with null handle");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(session->mu);

    const std::string image_path = to_std_string(env, jImagePath);
    const std::string prompt     = to_std_string(env, jPrompt);

    // Fresh conversation for every receipt: clear all KV-cache / memory state.
    llama_memory_clear(llama_get_memory(session->lctx), /*data=*/true);

    // --- load + preprocess the image ---
    // mtmd_helper_bitmap_init_from_file returns a wrapper struct by value now
    // (used to return mtmd_bitmap* directly), with an extra `placeholder`
    // param -- false here, we always have real image data, never a
    // count-only placeholder bitmap.
    mtmd_helper_bitmap_wrapper bitmap_wrapper =
            mtmd_helper_bitmap_init_from_file(session->mctx, image_path.c_str(), /*placeholder=*/false);
    mtmd_bitmap * bitmap = bitmap_wrapper.bitmap;
    if (bitmap == nullptr) {
        throw_runtime_exception(env, "Failed to load image: " + image_path);
        return nullptr;
    }

    // --- tokenize prompt (text mixed with the media marker) ---
    // The prompt must contain mtmd_default_marker() ("<__media__>") where the image goes;
    // the Kotlin side is responsible for that.
    // Zero-init, then set every field explicitly: mtmd_input_text is a plain
    // C struct with no default member initializers, so a bare declaration
    // leaves `text_len` as uninitialized stack garbage. mtmd.cpp uses it
    // directly (`input_text.assign(text->text, text->text_len)`), so this was
    // a real out-of-bounds/UB bug independent of the API-version drift below.
    mtmd_input_text input_text = {};
    input_text.text          = prompt.c_str();
    input_text.text_len      = prompt.size();
    input_text.add_special   = true;  // add BOS if the model uses one
    input_text.parse_special = true;  // parse <|im_start|> etc. as special tokens

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[1] = { bitmap };

    int32_t tok_res = mtmd_tokenize(session->mctx, chunks, &input_text, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        throw_runtime_exception(env, "mtmd_tokenize failed with code " + std::to_string(tok_res));
        return nullptr;
    }

    // --- evaluate all chunks (text tokens + image embeddings) ---
    llama_pos n_past = 0;
    int32_t eval_res = mtmd_helper_eval_chunks(
            session->mctx,
            session->lctx,
            chunks,
            /*n_past=*/0,
            /*seq_id=*/0,
            /*n_batch=*/session->n_batch,
            /*logits_last=*/true,
            &n_past);
    mtmd_input_chunks_free(chunks);
    if (eval_res != 0) {
        throw_runtime_exception(env, "mtmd_helper_eval_chunks failed with code " + std::to_string(eval_res));
        return nullptr;
    }

    // --- greedy sampling loop ---
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string output;
    output.reserve(4096);

    for (int i = 0; i < (int) maxTokens; ++i) {
        llama_token token = llama_sampler_sample(sampler, session->lctx, /*idx of last logits=*/-1);

        if (llama_vocab_is_eog(session->vocab, token)) {
            break;
        }

        char piece[256];
        int32_t n = llama_token_to_piece(session->vocab, token, piece, sizeof(piece),
                                         /*lstrip=*/0, /*special=*/true);
        if (n > 0) {
            output.append(piece, (size_t) n);
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(session->lctx, batch) != 0) {
            LOGE("llama_decode failed at generated token %d", i);
            break;
        }
    }

    llama_sampler_free(sampler);

    LOGI("extract: generated %zu bytes", output.size());
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_ledgerlens_app_LlamaBridge_unload(JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    auto * session = reinterpret_cast<Session *>(handle);
    if (session == nullptr) return;

    {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->mctx)  mtmd_free(session->mctx);
        if (session->lctx)  llama_free(session->lctx);
        if (session->model) llama_model_free(session->model);
        session->mctx  = nullptr;
        session->lctx  = nullptr;
        session->model = nullptr;
    }
    delete session;
    LOGI("unload: done");
}
