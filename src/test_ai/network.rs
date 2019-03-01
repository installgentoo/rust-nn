use std::{cell::RefCell, slice::Iter};

#[derive(Debug)]
struct NeuronRelu {
	weights: Vec<f64>,
	prev_dw: Vec<f64>,
	output: f64,
	d_output: f64,
}
impl NeuronRelu {
	const A: f64 = 0.01;

	fn boxed(weights: Vec<f64>) -> Box<Neuron> {
		let prev_dw = vec![0.; weights.len()];
		Box::new(NeuronRelu {
			weights,
			prev_dw,
			output: 0.,
			d_output: 0.,
		})
	}
}


#[derive(Debug)]
struct NeuronOut {
	weights: Vec<f64>,
	prev_dw: Vec<f64>,
}
impl NeuronOut {
	fn boxed(weights: Vec<f64>) -> Box<Neuron> {
		let prev_dw = vec![0.; weights.len()];
		Box::new(NeuronOut { weights, prev_dw })
	}
}



pub trait Neuron: std::fmt::Debug {
	fn run(&mut self, inputs: Iter<f64>) -> f64 {
		let (weights, _) = self.m_weights();
		let sum = weights.iter().skip(1).zip(inputs).fold(weights[0], |sum, (weight, input)| sum + weight * input);
		debug_assert!(sum.is_finite());

		self.activate(sum)
	}

	fn update(&mut self, d_out: f64, inertia: f64, inputs: Iter<f64>) {
		let (weights, prev_dws) = self.m_weights();
		let delta_w = inertia * prev_dws[0] + d_out;
		prev_dws[0] = delta_w;
		weights[0] += delta_w;
		weights.iter_mut().zip(prev_dws.iter_mut()).skip(1).zip(inputs).for_each(|((weight, prev_dw), input)| {
			let delta_w = inertia * *prev_dw + d_out * input;
			*prev_dw = delta_w;
			*weight += delta_w;
			debug_assert!(weight.is_finite());
		});
	}

	fn activate(&mut self, sum: f64) -> f64;
	fn derivative(&self) -> f64;
	fn weight(&self, at: usize) -> f64;
	fn m_weights(&mut self) -> (&mut Vec<f64>, &mut Vec<f64>);
	fn output(&self) -> f64;
}



impl Neuron for NeuronRelu {
	fn activate(&mut self, sum: f64) -> f64 {
		self.output = if sum >= 0. { sum } else { sum * NeuronRelu::A };
		self.d_output = if sum >= 0. { 1. } else { NeuronRelu::A };

		self.output
	}

	fn derivative(&self) -> f64 {
		self.d_output
	}

	fn weight(&self, at: usize) -> f64 {
		self.weights[at + 1]
	}

	fn m_weights(&mut self) -> (&mut Vec<f64>, &mut Vec<f64>) {
		(&mut self.weights, &mut self.prev_dw)
	}

	fn output(&self) -> f64 {
		self.output
	}
}


impl Neuron for NeuronOut {
	fn activate(&mut self, sum: f64) -> f64 {
		1. / (1. + (-sum).exp())
	}

	fn derivative(&self) -> f64 {
		1.
	}

	fn weight(&self, at: usize) -> f64 {
		self.weights[at + 1]
	}

	fn m_weights(&mut self) -> (&mut Vec<f64>, &mut Vec<f64>) {
		(&mut self.weights, &mut self.prev_dw)
	}

	fn output(&self) -> f64 {
		1.
	}
}



type NeuralLayer = Vec<RefCell<Box<dyn Neuron>>>;

pub struct NeuralNet {
	pub layers: Vec<NeuralLayer>,
	input: Vec<f64>,
}
impl NeuralNet {
	pub fn new(generator_func: &impl Fn() -> f64, layers_desc: &[u32]) -> NeuralNet {
		let layers = (1..layers_desc.len())
			.map(|i| {
				let current_layer_size = layers_desc[i];
				let prev_layer_size = layers_desc[i - 1];

				(0..current_layer_size)
					.map(|_| {
						let make_neuron = |new: &Fn(_) -> _| RefCell::new(new((0..1 + prev_layer_size).map(|_| generator_func()).collect()));

						if i + 1 < layers_desc.len() {
							make_neuron(&NeuronRelu::boxed)
						} else {
							make_neuron(&NeuronOut::boxed)
						}
					})
					.collect()
			})
			.collect();

		NeuralNet { layers, input: vec![] }
	}

	pub fn run(&mut self, input: Vec<f64>) -> Vec<f64> {
		self.input = input.clone();
		let mut input = input;
		let mut output: Vec<f64> = vec![];

		self.layers.iter().for_each(|layer| {
			output.resize(layer.len(), 0.);
			output.iter_mut().zip(layer).for_each(|(o, neuron)| *o = neuron.borrow_mut().run(input.iter()));
			std::mem::swap(&mut output, &mut input);
		});

		input
	}

	pub fn learn(&self, mut errors: Vec<f64>, learning_rate: f64, inertia: f64) {
		self.layers.iter().rev().skip(1).zip(self.layers.iter().rev()).for_each(|(prev_layer, layer)| {
			let error_gradient = (0..prev_layer.len())
				.map(|i| {
					layer.iter().zip(&errors).fold(0., |err, (neuron, error)| {
						let err = err + neuron.borrow().weight(i) * neuron.borrow().derivative() * error;
						debug_assert!(err.is_finite());
						err
					})
				})
				.collect();

			layer.iter().zip(&errors).for_each(|(neuron, error)| {
				let derivative = neuron.borrow().derivative();
				let inputs: Vec<f64> = prev_layer.iter().map(|n| n.borrow().output()).collect();
				neuron.borrow_mut().update(learning_rate * derivative * error, inertia, inputs.iter());
			});

			errors = error_gradient;
		});

		self.layers[0].iter().zip(&errors).for_each(|(neuron, error)| {
			let derivative = neuron.borrow().derivative();
			neuron.borrow_mut().update(learning_rate * derivative * error, inertia, self.input.iter());
		});
	}
}
