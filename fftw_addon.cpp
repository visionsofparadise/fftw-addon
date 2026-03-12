#include <napi.h>
#include <fftw3.h>
#include <unordered_map>
#include <mutex>
#include <cstring>

static std::unordered_map<int, fftwf_plan> fwd_plans;
static std::unordered_map<int, fftwf_plan> inv_plans;
static std::mutex plan_mutex;

static fftwf_plan get_fwd_plan(int fft_size) {
	std::lock_guard<std::mutex> lock(plan_mutex);
	auto it = fwd_plans.find(fft_size);
	if (it != fwd_plans.end()) return it->second;

	int complex_size = fft_size / 2 + 1;
	float* in = fftwf_alloc_real(fft_size);
	fftwf_complex* out = fftwf_alloc_complex(complex_size);
	fftwf_plan p = fftwf_plan_dft_r2c_1d(fft_size, in, out, FFTW_MEASURE);
	fftwf_free(in);
	fftwf_free(out);
	fwd_plans[fft_size] = p;
	return p;
}

static fftwf_plan get_inv_plan(int fft_size) {
	std::lock_guard<std::mutex> lock(plan_mutex);
	auto it = inv_plans.find(fft_size);
	if (it != inv_plans.end()) return it->second;

	int complex_size = fft_size / 2 + 1;
	fftwf_complex* in = fftwf_alloc_complex(complex_size);
	float* out = fftwf_alloc_real(fft_size);
	fftwf_plan p = fftwf_plan_dft_c2r_1d(fft_size, in, out, FFTW_MEASURE);
	fftwf_free(in);
	fftwf_free(out);
	inv_plans[fft_size] = p;
	return p;
}

// batchFft(input: Float32Array, fftSize: number, batchCount: number) => { re: Float32Array, im: Float32Array }
Napi::Value BatchFft(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 3) {
		Napi::TypeError::New(env, "Expected 3 arguments: input, fftSize, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array input = info[0].As<Napi::Float32Array>();
	int fft_size = info[1].As<Napi::Number>().Int32Value();
	int batch_count = info[2].As<Napi::Number>().Int32Value();
	int complex_size = fft_size / 2 + 1;

	if ((int)input.ElementLength() < fft_size * batch_count) {
		Napi::RangeError::New(env, "Input buffer too small for fftSize * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	fftwf_plan plan = get_fwd_plan(fft_size);

	Napi::Float32Array re = Napi::Float32Array::New(env, complex_size * batch_count);
	Napi::Float32Array im = Napi::Float32Array::New(env, complex_size * batch_count);

	float* in_buf = fftwf_alloc_real(fft_size);
	fftwf_complex* out_buf = fftwf_alloc_complex(complex_size);

	float* input_data = input.Data();

	for (int b = 0; b < batch_count; b++) {
		std::memcpy(in_buf, input_data + b * fft_size, fft_size * sizeof(float));
		fftwf_execute_dft_r2c(plan, in_buf, out_buf);

		for (int i = 0; i < complex_size; i++) {
			re[b * complex_size + i] = out_buf[i][0];
			im[b * complex_size + i] = out_buf[i][1];
		}
	}

	fftwf_free(in_buf);
	fftwf_free(out_buf);

	Napi::Object result = Napi::Object::New(env);
	result.Set("re", re);
	result.Set("im", im);
	return result;
}

// batchIfft(re: Float32Array, im: Float32Array, fftSize: number, batchCount: number) => Float32Array
Napi::Value BatchIfft(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 4) {
		Napi::TypeError::New(env, "Expected 4 arguments: re, im, fftSize, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array re = info[0].As<Napi::Float32Array>();
	Napi::Float32Array im = info[1].As<Napi::Float32Array>();
	int fft_size = info[2].As<Napi::Number>().Int32Value();
	int batch_count = info[3].As<Napi::Number>().Int32Value();
	int complex_size = fft_size / 2 + 1;

	fftwf_plan plan = get_inv_plan(fft_size);

	Napi::Float32Array output = Napi::Float32Array::New(env, fft_size * batch_count);

	fftwf_complex* in_buf = fftwf_alloc_complex(complex_size);
	float* out_buf = fftwf_alloc_real(fft_size);

	float scale = 1.0f / fft_size;

	for (int b = 0; b < batch_count; b++) {
		for (int i = 0; i < complex_size; i++) {
			in_buf[i][0] = re[b * complex_size + i];
			in_buf[i][1] = im[b * complex_size + i];
		}
		fftwf_execute_dft_c2r(plan, in_buf, out_buf);

		for (int i = 0; i < fft_size; i++) {
			output[b * fft_size + i] = out_buf[i] * scale;
		}
	}

	fftwf_free(in_buf);
	fftwf_free(out_buf);

	return output;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
	exports.Set("batchFft", Napi::Function::New(env, BatchFft));
	exports.Set("batchIfft", Napi::Function::New(env, BatchIfft));
	return exports;
}

NODE_API_MODULE(fftw_addon, Init)
