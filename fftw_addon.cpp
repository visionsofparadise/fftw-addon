#include <napi.h>
#include <fftw3.h>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cstring>

static std::unordered_map<int, fftwf_plan> fwd_plans;
static std::unordered_map<int, fftwf_plan> inv_plans;
static std::unordered_map<uint64_t, fftwf_plan> fwd_2d_plans;
static std::unordered_map<uint64_t, fftwf_plan> inv_2d_plans;
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

static inline uint64_t shape_key_2d(int rows, int cols) {
	return (static_cast<uint64_t>(static_cast<uint32_t>(rows)) << 32) |
	       static_cast<uint64_t>(static_cast<uint32_t>(cols));
}

static fftwf_plan get_fwd_2d_plan(int rows, int cols) {
	std::lock_guard<std::mutex> lock(plan_mutex);
	uint64_t key = shape_key_2d(rows, cols);
	auto it = fwd_2d_plans.find(key);
	if (it != fwd_2d_plans.end()) return it->second;

	int complex_cols = cols / 2 + 1;
	float* in = fftwf_alloc_real(rows * cols);
	fftwf_complex* out = fftwf_alloc_complex(rows * complex_cols);
	fftwf_plan p = fftwf_plan_dft_r2c_2d(rows, cols, in, out, FFTW_MEASURE);
	fftwf_free(in);
	fftwf_free(out);
	fwd_2d_plans[key] = p;
	return p;
}

static fftwf_plan get_inv_2d_plan(int rows, int cols) {
	std::lock_guard<std::mutex> lock(plan_mutex);
	uint64_t key = shape_key_2d(rows, cols);
	auto it = inv_2d_plans.find(key);
	if (it != inv_2d_plans.end()) return it->second;

	int complex_cols = cols / 2 + 1;
	fftwf_complex* in = fftwf_alloc_complex(rows * complex_cols);
	float* out = fftwf_alloc_real(rows * cols);
	fftwf_plan p = fftwf_plan_dft_c2r_2d(rows, cols, in, out, FFTW_MEASURE);
	fftwf_free(in);
	fftwf_free(out);
	inv_2d_plans[key] = p;
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

// batchFftInto(input: Float32Array, outRe: Float32Array, outIm: Float32Array, fftSize: number, batchCount: number) => undefined
// Writes into caller-owned outRe/outIm. No allocation, no return value.
Napi::Value BatchFftInto(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 5) {
		Napi::TypeError::New(env, "Expected 5 arguments: input, outRe, outIm, fftSize, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array input = info[0].As<Napi::Float32Array>();
	Napi::Float32Array out_re = info[1].As<Napi::Float32Array>();
	Napi::Float32Array out_im = info[2].As<Napi::Float32Array>();
	int fft_size = info[3].As<Napi::Number>().Int32Value();
	int batch_count = info[4].As<Napi::Number>().Int32Value();
	int complex_size = fft_size / 2 + 1;

	if ((int)input.ElementLength() < fft_size * batch_count) {
		Napi::RangeError::New(env, "Input buffer too small for fftSize * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	if ((int)out_re.ElementLength() < complex_size * batch_count ||
	    (int)out_im.ElementLength() < complex_size * batch_count) {
		Napi::RangeError::New(env, "outRe/outIm buffers too small for (fftSize / 2 + 1) * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	fftwf_plan plan = get_fwd_plan(fft_size);

	float* in_buf = fftwf_alloc_real(fft_size);
	fftwf_complex* out_buf = fftwf_alloc_complex(complex_size);

	float* input_data = input.Data();
	float* re_data = out_re.Data();
	float* im_data = out_im.Data();

	for (int b = 0; b < batch_count; b++) {
		std::memcpy(in_buf, input_data + b * fft_size, fft_size * sizeof(float));
		fftwf_execute_dft_r2c(plan, in_buf, out_buf);

		int batch_offset = b * complex_size;
		for (int i = 0; i < complex_size; i++) {
			re_data[batch_offset + i] = out_buf[i][0];
			im_data[batch_offset + i] = out_buf[i][1];
		}
	}

	fftwf_free(in_buf);
	fftwf_free(out_buf);

	return env.Undefined();
}

// batchIfftInto(re: Float32Array, im: Float32Array, outTime: Float32Array, fftSize: number, batchCount: number) => undefined
// Writes into caller-owned outTime. No allocation, no return value.
Napi::Value BatchIfftInto(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 5) {
		Napi::TypeError::New(env, "Expected 5 arguments: re, im, outTime, fftSize, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array re = info[0].As<Napi::Float32Array>();
	Napi::Float32Array im = info[1].As<Napi::Float32Array>();
	Napi::Float32Array out_time = info[2].As<Napi::Float32Array>();
	int fft_size = info[3].As<Napi::Number>().Int32Value();
	int batch_count = info[4].As<Napi::Number>().Int32Value();
	int complex_size = fft_size / 2 + 1;

	if ((int)re.ElementLength() < complex_size * batch_count ||
	    (int)im.ElementLength() < complex_size * batch_count) {
		Napi::RangeError::New(env, "re/im buffers too small for (fftSize / 2 + 1) * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	if ((int)out_time.ElementLength() < fft_size * batch_count) {
		Napi::RangeError::New(env, "outTime buffer too small for fftSize * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	fftwf_plan plan = get_inv_plan(fft_size);

	fftwf_complex* in_buf = fftwf_alloc_complex(complex_size);
	float* out_buf = fftwf_alloc_real(fft_size);

	float scale = 1.0f / fft_size;

	float* out_data = out_time.Data();

	for (int b = 0; b < batch_count; b++) {
		int batch_offset = b * complex_size;
		for (int i = 0; i < complex_size; i++) {
			in_buf[i][0] = re[batch_offset + i];
			in_buf[i][1] = im[batch_offset + i];
		}
		fftwf_execute_dft_c2r(plan, in_buf, out_buf);

		int out_offset = b * fft_size;
		for (int i = 0; i < fft_size; i++) {
			out_data[out_offset + i] = out_buf[i] * scale;
		}
	}

	fftwf_free(in_buf);
	fftwf_free(out_buf);

	return env.Undefined();
}

// batchFft2D(input: Float32Array, rows: number, cols: number, batchCount: number) => { re: Float32Array, im: Float32Array }
// Input: rows * cols * batchCount real samples, block-major, row-major within each block (row index outer, col index inner).
// Output: rows * (cols / 2 + 1) * batchCount complex samples split into parallel re/im arrays, same block-major ordering.
Napi::Value BatchFft2D(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 4) {
		Napi::TypeError::New(env, "Expected 4 arguments: input, rows, cols, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array input = info[0].As<Napi::Float32Array>();
	int rows = info[1].As<Napi::Number>().Int32Value();
	int cols = info[2].As<Napi::Number>().Int32Value();
	int batch_count = info[3].As<Napi::Number>().Int32Value();

	if (rows <= 0 || cols <= 0 || batch_count < 0) {
		Napi::RangeError::New(env, "rows, cols must be positive and batchCount non-negative").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	int real_block = rows * cols;
	int complex_cols = cols / 2 + 1;
	int complex_block = rows * complex_cols;

	if ((int)input.ElementLength() < real_block * batch_count) {
		Napi::RangeError::New(env, "Input buffer too small for rows * cols * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	fftwf_plan plan = get_fwd_2d_plan(rows, cols);

	Napi::Float32Array re = Napi::Float32Array::New(env, complex_block * batch_count);
	Napi::Float32Array im = Napi::Float32Array::New(env, complex_block * batch_count);

	float* in_buf = fftwf_alloc_real(real_block);
	fftwf_complex* out_buf = fftwf_alloc_complex(complex_block);

	float* input_data = input.Data();

	for (int b = 0; b < batch_count; b++) {
		std::memcpy(in_buf, input_data + b * real_block, real_block * sizeof(float));
		fftwf_execute_dft_r2c(plan, in_buf, out_buf);

		int batch_offset = b * complex_block;
		for (int i = 0; i < complex_block; i++) {
			re[batch_offset + i] = out_buf[i][0];
			im[batch_offset + i] = out_buf[i][1];
		}
	}

	fftwf_free(in_buf);
	fftwf_free(out_buf);

	Napi::Object result = Napi::Object::New(env);
	result.Set("re", re);
	result.Set("im", im);
	return result;
}

// batchIfft2D(re: Float32Array, im: Float32Array, rows: number, cols: number, batchCount: number) => Float32Array
// Inputs: rows * (cols / 2 + 1) * batchCount complex samples split into parallel re/im arrays.
// Output: rows * cols * batchCount real samples, scaled by 1 / (rows * cols) to invert FFTW's unscaled forward transform.
Napi::Value BatchIfft2D(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 5) {
		Napi::TypeError::New(env, "Expected 5 arguments: re, im, rows, cols, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array re = info[0].As<Napi::Float32Array>();
	Napi::Float32Array im = info[1].As<Napi::Float32Array>();
	int rows = info[2].As<Napi::Number>().Int32Value();
	int cols = info[3].As<Napi::Number>().Int32Value();
	int batch_count = info[4].As<Napi::Number>().Int32Value();

	if (rows <= 0 || cols <= 0 || batch_count < 0) {
		Napi::RangeError::New(env, "rows, cols must be positive and batchCount non-negative").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	int real_block = rows * cols;
	int complex_cols = cols / 2 + 1;
	int complex_block = rows * complex_cols;

	if ((int)re.ElementLength() < complex_block * batch_count ||
	    (int)im.ElementLength() < complex_block * batch_count) {
		Napi::RangeError::New(env, "re/im buffers too small for rows * (cols / 2 + 1) * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	fftwf_plan plan = get_inv_2d_plan(rows, cols);

	Napi::Float32Array output = Napi::Float32Array::New(env, real_block * batch_count);

	fftwf_complex* in_buf = fftwf_alloc_complex(complex_block);
	float* out_buf = fftwf_alloc_real(real_block);

	float scale = 1.0f / static_cast<float>(real_block);

	for (int b = 0; b < batch_count; b++) {
		int batch_offset = b * complex_block;
		for (int i = 0; i < complex_block; i++) {
			in_buf[i][0] = re[batch_offset + i];
			in_buf[i][1] = im[batch_offset + i];
		}
		fftwf_execute_dft_c2r(plan, in_buf, out_buf);

		int out_offset = b * real_block;
		for (int i = 0; i < real_block; i++) {
			output[out_offset + i] = out_buf[i] * scale;
		}
	}

	fftwf_free(in_buf);
	fftwf_free(out_buf);

	return output;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
	exports.Set("batchFft", Napi::Function::New(env, BatchFft));
	exports.Set("batchIfft", Napi::Function::New(env, BatchIfft));
	exports.Set("batchFftInto", Napi::Function::New(env, BatchFftInto));
	exports.Set("batchIfftInto", Napi::Function::New(env, BatchIfftInto));
	exports.Set("batchFft2D", Napi::Function::New(env, BatchFft2D));
	exports.Set("batchIfft2D", Napi::Function::New(env, BatchIfft2D));
	return exports;
}

NODE_API_MODULE(fftw_addon, Init)
