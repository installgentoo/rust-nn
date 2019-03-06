mod layers;
mod neurons;

use std::{cell::RefCell};
use layers::{Layer, LeakyRleu, Out};

pub struct NeuralNet {
	pub layers: Vec<RefCell<Box<dyn Layer>>>,
	input: Vec<f64>,
}
impl NeuralNet {
	pub fn new(layers_desc: &[u32], generator_func: &impl Fn() -> f64) -> NeuralNet {
		let layers = (1..layers_desc.len())
			.map(|i| {
				let current_layer_size = layers_desc[i];
				let prev_layer_size = layers_desc[i - 1];

				RefCell::new(if i + 1 < layers_desc.len() {
					LeakyRleu::boxed(prev_layer_size, current_layer_size, generator_func)
				} else {
					Out::boxed(prev_layer_size, current_layer_size, generator_func)
				})
			})
			.collect();

		NeuralNet { layers, input: vec![] }
	}

	pub fn run(&mut self, input: Vec<f64>) -> Vec<f64> {
		self.input = input.clone();
		let mut input = input;
		let mut output: Vec<f64> = vec![];

		self.layers.iter().for_each(|layer| {
			layer.borrow_mut().run(&mut output, &input);
			std::mem::swap(&mut output, &mut input);
		});

		input
	}

	pub fn learn(&self, mut errors: Vec<f64>, learning_rate: f64, inertia: f64) {
		let mut error_gradient: Vec<f64> = vec![];

		self.layers.iter().rev().skip(1).zip(self.layers.iter().rev()).for_each(|(prev_layer, layer)| {
			layer.borrow().gen_errors(&mut error_gradient, &errors, prev_layer.borrow().len());

			layer.borrow_mut().learn(&errors, &prev_layer.borrow().outputs(), learning_rate, inertia);

			std::mem::swap(&mut errors, &mut error_gradient);
		});

		self.layers[0].borrow_mut().learn(&errors, &self.input, learning_rate, inertia);
	}
}
