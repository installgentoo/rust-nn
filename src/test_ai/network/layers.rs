use super::neurons;

#[derive(Debug)]
pub struct Out {
	neurons: Vec<neurons::Out>,
}
impl Out {
	pub fn boxed(prev_layer_size: u32, size: u32, generator_func: &impl Fn() -> f64) -> Box<Layer> {
		Box::new(Out {
			neurons: (0..size).map(|_| neurons::Out::new((0..1 + prev_layer_size).map(|_| generator_func()).collect())).collect(),
		})
	}
}

#[derive(Debug)]
pub struct LeakyRleu {
	neurons: Vec<neurons::LeakyRelu>,
}
impl LeakyRleu {
	pub fn boxed(prev_layer_size: u32, size: u32, generator_func: &impl Fn() -> f64) -> Box<Layer> {
		Box::new(LeakyRleu {
			neurons: (0..size)
				.map(|_| neurons::LeakyRelu::new((0..1 + prev_layer_size).map(|_| generator_func()).collect()))
				.collect(),
		})
	}
}

pub trait Layer: std::fmt::Debug {
	fn run(&mut self, output: &mut Vec<f64>, input: &[f64]);
	fn gen_errors(&self, gradient: &mut Vec<f64>, errors: &[f64], inputs_number: usize);
	fn learn(&mut self, errors: &[f64], prev_layer: &[f64], learning_rate: f64, inertia: f64);
	fn outputs(&self) -> Vec<f64>;
	fn len(&self) -> usize;
}

mod layer_shared {
	use super::*;

	pub fn run<T: neurons::Neuron>(layer: &mut Vec<T>, output: &mut Vec<f64>, input: &[f64]) {
		output.resize(layer.len(), 0.);
		output.iter_mut().zip(layer).for_each(|(o, neuron)| *o = neuron.run(input));
	}

	pub fn gen_errors<T: neurons::Neuron>(layer: &[T], gradient: &mut Vec<f64>, errors: &[f64], inputs_number: usize) {
		let neurons_errors = layer.iter().zip(errors);
		gradient.resize(inputs_number as usize, 0.);

		(0..gradient.len()).for_each(|i| {
			gradient[i] = neurons_errors.clone().fold(0., |err, (neuron, error)| {
				let err = err + neuron.weight(i) * neuron.derivative() * error;
				debug_assert!(err.is_finite());
				err
			});
		});
	}

	pub fn learn<N: neurons::Neuron>(layer: &mut [N], errors: &[f64], prev_layer: &[f64], learning_rate: f64, inertia: f64) {
		layer.iter_mut().zip(errors).for_each(|(neuron, error)| {
			neuron.update(learning_rate * neuron.derivative() * error, inertia, prev_layer);
		});
	}
}

impl Layer for LeakyRleu {
	fn run(&mut self, output: &mut Vec<f64>, input: &[f64]) {
		layer_shared::run(&mut self.neurons, output, input);
	}
	fn gen_errors(&self, gradient: &mut Vec<f64>, errors: &[f64], inputs_number: usize) {
		layer_shared::gen_errors(&self.neurons, gradient, errors, inputs_number);
	}
	fn learn(&mut self, errors: &[f64], prev_layer: &[f64], learning_rate: f64, inertia: f64) {
		layer_shared::learn(&mut self.neurons, errors, prev_layer, learning_rate, inertia);
	}
	fn len(&self) -> usize {
		self.neurons.len()
	}
	fn outputs(&self) -> Vec<f64> {
		self.neurons.iter().map(|n| n.output()).collect()
	}
}

impl Layer for Out {
	fn run(&mut self, output: &mut Vec<f64>, input: &[f64]) {
		layer_shared::run(&mut self.neurons, output, input);
	}
	fn gen_errors(&self, gradient: &mut Vec<f64>, errors: &[f64], inputs_number: usize) {
		layer_shared::gen_errors(&self.neurons, gradient, errors, inputs_number);
	}
	fn learn(&mut self, errors: &[f64], prev_layer: &[f64], learning_rate: f64, inertia: f64) {
		layer_shared::learn(&mut self.neurons, errors, prev_layer, learning_rate, inertia);
	}
	fn len(&self) -> usize {
		self.neurons.len()
	}
	fn outputs(&self) -> Vec<f64> {
		panic!()
	}
}
