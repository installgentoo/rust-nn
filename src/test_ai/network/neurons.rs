#[derive(Debug)]
pub struct LeakyRelu {
	weights: Vec<f64>,
	prev_dws: Vec<f64>,
	output: f64,
	d_output: f64,
}
impl LeakyRelu {
	const A: f64 = 0.01;

	pub fn new(weights: Vec<f64>) -> LeakyRelu {
		let prev_dws = vec![0.; weights.len()];
		LeakyRelu {
			weights,
			prev_dws,
			output: 0.,
			d_output: 0.,
		}
	}
	pub fn output(&self) -> f64 {
		self.output
	}
}

#[derive(Debug)]
pub struct Out {
	weights: Vec<f64>,
	prev_dws: Vec<f64>,
}
impl Out {
	pub fn new(weights: Vec<f64>) -> Out {
		let prev_dws = vec![0.; weights.len()];
		Out { weights, prev_dws }
	}
}

pub trait Neuron {
	fn run(&mut self, inputs: &[f64]) -> f64;
	fn update(&mut self, d_out: f64, inertia: f64, inputs: &[f64]);
	fn derivative(&self) -> f64;
	fn weight(&self, at: usize) -> f64;
}

mod neuron_shared {
	pub fn sum(weights: &[f64], inputs: &[f64]) -> f64 {
		let sum = weights.iter().skip(1).zip(inputs).fold(weights[0], |sum, (weight, input)| sum + weight * input);
		debug_assert!(sum.is_finite());
		sum
	}

	pub fn update(weights: &mut [f64], prev_dws: &mut [f64], d_out: f64, inertia: f64, inputs: &[f64]) {
		let delta_w = inertia * prev_dws[0] + d_out;
		prev_dws[0] = delta_w;
		weights[0] += delta_w;
		weights.iter_mut().zip(prev_dws).skip(1).zip(inputs).for_each(|((weight, prev_dw), input)| {
			let delta_w = inertia * *prev_dw + d_out * input;
			*prev_dw = delta_w;
			*weight += delta_w;
			debug_assert!(weight.is_finite());
		});
	}
}

impl Neuron for LeakyRelu {
	fn run(&mut self, inputs: &[f64]) -> f64 {
		let sum = neuron_shared::sum(&self.weights, inputs);

		self.output = if sum >= 0. { sum } else { sum * LeakyRelu::A };
		self.d_output = if sum >= 0. { 1. } else { LeakyRelu::A };

		self.output
	}

	fn derivative(&self) -> f64 {
		self.d_output
	}

	fn update(&mut self, d_out: f64, inertia: f64, inputs: &[f64]) {
		neuron_shared::update(&mut self.weights, &mut self.prev_dws, d_out, inertia, inputs);
	}
	fn weight(&self, at: usize) -> f64 {
		self.weights[at + 1]
	}
}

impl Neuron for Out {
	fn run(&mut self, inputs: &[f64]) -> f64 {
		let sum = neuron_shared::sum(&self.weights, inputs);

		1. / (1. + (-sum).exp())
	}

	fn derivative(&self) -> f64 {
		1.
	}

	fn update(&mut self, d_out: f64, inertia: f64, inputs: &[f64]) {
		neuron_shared::update(&mut self.weights, &mut self.prev_dws, d_out, inertia, inputs);
	}
	fn weight(&self, at: usize) -> f64 {
		self.weights[at + 1]
	}
}
