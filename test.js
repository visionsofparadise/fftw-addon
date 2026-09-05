const path = require("path");
const addon = require(path.join(__dirname, "build/Release/fftw_addon.node"));

function maxError(a, b) {
	let max = 0;
	for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
	return max;
}

let failed = 0;

function test(name, fn) {
	try {
		fn();
		console.log(`PASS: ${name}`);
	} catch (e) {
		console.log(`FAIL: ${name} — ${e.message}`);
		failed++;
	}
}

function assert(cond, msg) {
	if (!cond) throw new Error(msg);
}

test("batchFft returns correct shape", () => {
	const fftSize = 1024;
	const input = new Float32Array(fftSize).fill(0);
	input[0] = 1; // impulse
	const { re, im } = addon.batchFft(input, fftSize, 1);
	assert(re.length === fftSize / 2 + 1, `re length ${re.length}`);
	assert(im.length === fftSize / 2 + 1, `im length ${im.length}`);
	// impulse -> flat spectrum, all re = 1, all im = 0
	assert(Math.abs(re[0] - 1) < 1e-5, `DC bin re=${re[0]}`);
	assert(Math.abs(im[0]) < 1e-5, `DC bin im=${im[0]}`);
});

test("round-trip fft -> ifft", () => {
	const fftSize = 1024;
	const batchCount = 4;
	const input = new Float32Array(fftSize * batchCount);
	for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
	const { re, im } = addon.batchFft(input, fftSize, batchCount);
	const output = addon.batchIfft(re, im, fftSize, batchCount);
	const err = maxError(input, output);
	assert(err < 1e-5, `error ${err}`);
	console.log(`  max error: ${err.toExponential(2)}`);
});

test("multiple FFT sizes", () => {
	for (const fftSize of [256, 512, 2048, 4096]) {
		const input = new Float32Array(fftSize);
		for (let i = 0; i < fftSize; i++) input[i] = Math.sin((2 * Math.PI * i) / fftSize);
		const { re, im } = addon.batchFft(input, fftSize, 1);
		const output = addon.batchIfft(re, im, fftSize, 1);
		const err = maxError(input, output);
		assert(err < 1e-5, `fftSize=${fftSize} error=${err}`);
	}
});

test("batchFftInto round-trip with caller-owned buffers", () => {
	const fftSize = 1024;
	const batchCount = 4;
	const complexSize = fftSize / 2 + 1;
	const input = new Float32Array(fftSize * batchCount);
	for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
	const re = new Float32Array(complexSize * batchCount);
	const im = new Float32Array(complexSize * batchCount);
	const result = addon.batchFftInto(input, re, im, fftSize, batchCount);
	assert(result === undefined, `expected undefined return, got ${result}`);
	const output = addon.batchIfft(re, im, fftSize, batchCount);
	const err = maxError(input, output);
	assert(err < 1e-5, `error ${err}`);
	console.log(`  max error: ${err.toExponential(2)}`);
});

test("batchIfftInto round-trip with caller-owned buffer", () => {
	const fftSize = 1024;
	const batchCount = 4;
	const input = new Float32Array(fftSize * batchCount);
	for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
	const { re, im } = addon.batchFft(input, fftSize, batchCount);
	const output = new Float32Array(fftSize * batchCount);
	const result = addon.batchIfftInto(re, im, output, fftSize, batchCount);
	assert(result === undefined, `expected undefined return, got ${result}`);
	const err = maxError(input, output);
	assert(err < 1e-5, `error ${err}`);
	console.log(`  max error: ${err.toExponential(2)}`);
});

test("batchFftInto + batchIfftInto end-to-end round-trip", () => {
	const fftSize = 2048;
	const batchCount = 3;
	const complexSize = fftSize / 2 + 1;
	const input = new Float32Array(fftSize * batchCount);
	for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.1) * ((i % 7) - 3);
	const re = new Float32Array(complexSize * batchCount);
	const im = new Float32Array(complexSize * batchCount);
	addon.batchFftInto(input, re, im, fftSize, batchCount);
	const output = new Float32Array(fftSize * batchCount);
	addon.batchIfftInto(re, im, output, fftSize, batchCount);
	const err = maxError(input, output);
	assert(err < 1e-5, `error ${err}`);
	console.log(`  max error: ${err.toExponential(2)}`);
});

test("batchFftInto throws RangeError for undersized output buffers", () => {
	const fftSize = 256;
	const batchCount = 2;
	const complexSize = fftSize / 2 + 1;
	const input = new Float32Array(fftSize * batchCount);
	const reSmall = new Float32Array(complexSize * batchCount - 1);
	const im = new Float32Array(complexSize * batchCount);
	let threw = false;
	try {
		addon.batchFftInto(input, reSmall, im, fftSize, batchCount);
	} catch (e) {
		threw = true;
		assert(e instanceof RangeError, `expected RangeError, got ${e.constructor.name}`);
	}
	assert(threw, "expected throw for undersized outRe");
});

test("batchIfftInto throws RangeError for undersized output buffer", () => {
	const fftSize = 256;
	const batchCount = 2;
	const complexSize = fftSize / 2 + 1;
	const re = new Float32Array(complexSize * batchCount);
	const im = new Float32Array(complexSize * batchCount);
	const outSmall = new Float32Array(fftSize * batchCount - 1);
	let threw = false;
	try {
		addon.batchIfftInto(re, im, outSmall, fftSize, batchCount);
	} catch (e) {
		threw = true;
		assert(e instanceof RangeError, `expected RangeError, got ${e.constructor.name}`);
	}
	assert(threw, "expected throw for undersized outTime");
});

test("batchFft2D returns correct shape", () => {
	const rows = 4;
	const cols = 4;
	const input = new Float32Array(rows * cols).fill(0);
	input[0] = 1; // impulse at (0, 0)
	const { re, im } = addon.batchFft2D(input, rows, cols, 1);
	const complexCols = cols / 2 + 1;
	assert(re.length === rows * complexCols, `re length ${re.length}`);
	assert(im.length === rows * complexCols, `im length ${im.length}`);
	// Impulse at (0,0) -> flat 2D spectrum: all re = 1, all im = 0.
	for (let i = 0; i < re.length; i++) {
		assert(Math.abs(re[i] - 1) < 1e-5, `bin ${i} re=${re[i]}`);
		assert(Math.abs(im[i]) < 1e-5, `bin ${i} im=${im[i]}`);
	}
});

test("batchFft2D round-trip 4x4 batched", () => {
	const rows = 4;
	const cols = 4;
	const batchCount = 2;
	const input = new Float32Array(rows * cols * batchCount);
	for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
	const { re, im } = addon.batchFft2D(input, rows, cols, batchCount);
	const output = addon.batchIfft2D(re, im, rows, cols, batchCount);
	const err = maxError(input, output);
	assert(err < 1e-5, `error ${err}`);
	console.log(`  max error: ${err.toExponential(2)}`);
});

test("batchFft2D non-square rows=16 cols=32 (DFTT shape)", () => {
	// Matches the engineering-side DFTT block layout: rows = blockTime = 16,
	// cols = blockFreq = 32, block-major and row-major (time outer, freq inner)
	// inside each block. Non-square deliberately to catch any rows/cols swap
	// between addon and caller.
	const rows = 16;
	const cols = 32;
	const batchCount = 100;
	const input = new Float32Array(rows * cols * batchCount);
	for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
	const { re, im } = addon.batchFft2D(input, rows, cols, batchCount);
	const complexCols = cols / 2 + 1;
	assert(
		re.length === rows * complexCols * batchCount,
		`re length ${re.length}, expected ${rows * complexCols * batchCount}`,
	);
	assert(
		im.length === rows * complexCols * batchCount,
		`im length ${im.length}, expected ${rows * complexCols * batchCount}`,
	);
	const output = addon.batchIfft2D(re, im, rows, cols, batchCount);
	assert(output.length === rows * cols * batchCount, `output length ${output.length}`);
	const err = maxError(input, output);
	assert(err < 1e-5, `error ${err}`);
	console.log(`  max error: ${err.toExponential(2)}`);
});

test("batchFft2D DC block across multiple 2D sizes", () => {
	// Each batch element is a constant value; the DFT is an impulse at (0,0)
	// with magnitude rows * cols, zero elsewhere. Validates the 2D plan cache
	// handles repeat shapes and distinct shapes in sequence without interference.
	for (const [rows, cols] of [
		[8, 8],
		[16, 32],
		[32, 16],
		[4, 8],
	]) {
		const batchCount = 3;
		const input = new Float32Array(rows * cols * batchCount);
		for (let b = 0; b < batchCount; b++) {
			const v = b + 1;
			for (let i = 0; i < rows * cols; i++) input[b * rows * cols + i] = v;
		}
		const { re, im } = addon.batchFft2D(input, rows, cols, batchCount);
		const complexCols = cols / 2 + 1;
		const complexBlock = rows * complexCols;
		for (let b = 0; b < batchCount; b++) {
			const v = b + 1;
			const expectedDc = v * rows * cols;
			assert(
				Math.abs(re[b * complexBlock] - expectedDc) < 1e-3,
				`rows=${rows} cols=${cols} b=${b} DC re=${re[b * complexBlock]}, expected ${expectedDc}`,
			);
			assert(
				Math.abs(im[b * complexBlock]) < 1e-3,
				`rows=${rows} cols=${cols} b=${b} DC im=${im[b * complexBlock]}`,
			);
		}
		const output = addon.batchIfft2D(re, im, rows, cols, batchCount);
		const err = maxError(input, output);
		assert(err < 1e-4, `rows=${rows} cols=${cols} error=${err}`);
	}
});

process.exit(failed);
