#![warn(clippy::all)]
#![allow(clippy::range_plus_one)]

mod network;

use rand_xorshift::XorShiftRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use network::NeuralNet;

fn main() {
	let rng = RefCell::new(XorShiftRng::from_rng(rand::thread_rng()).unwrap());

	let mut net = NeuralNet::new(&|| rng.borrow_mut().gen_range(0., 1.), &[2, 8, 2]);

	println!(
		"Initialized new network {}",
		format!("{:?}", net.layers)
			.replace("RefCell { value: ", "\n")
			.replace("{ weights: ", "")
			.replace("], [", "\n\n")
			.replace(" } }", "")
	);

	let dataset: Vec<_> = (0..100_000)
		.map(|_| {
			let mut rng = rng.borrow_mut();
			let (x, y): (f64, f64) = (rng.gen_range(0., 1.), rng.gen_range(0., 1.));

			let d = (x * x + y * y).sqrt();
			let expected = if (d > 0.3) && (d < 0.7) { vec![1., 0.] } else { vec![0., 1.] };

			(x, y, expected)
		})
		.collect();

	(0..1000).for_each(|i| {
		(0..10).for_each(|_| {
			dataset.iter().skip(i * 10).take(10).for_each(|data| {
				let (x, y, expected) = data;

				let result = net.run(vec![*x, *y]);

				let error: Vec<_> = expected.iter().zip(&result).map(|(exp, res)| exp - res).collect();
				let err = error.iter().fold(0., |sum, v| sum + v * v);
				net.learn(error, 0.01, 0.1);

				println!("x:{:.*} y:{:.*}, r:{:.*?} e:{:?}, err:{:.*}", 2, x, 2, y, 2, result, expected, 1, err);
			});
		});
		println!("\n\n\n................\nfinished batch\n................\n\n\n");
	});

	let total_err = dataset.iter().rev().take(1000).fold(0., |total_err, data| {
		let (x, y, expected) = data;

		let result = net.run(vec![*x, *y]);

		let error: Vec<_> = expected.iter().zip(&result).map(|(exp, res)| exp - res).collect();
		let err = error.iter().fold(0., |sum, v| sum + v * v);

		println!("x:{:.*} y:{:.*}, r:{:.*?} e:{:?}, err:{:.*}", 2, x, 2, y, 2, result, expected, 1, err);
		total_err + err
	});

	println!("\n\n.............\nTotal test error: {}\n.............\n\n", total_err);

	println!(
		"Network after training {}",
		format!("{:?}", net.layers)
			.replace("RefCell { value: ", "\n")
			.replace("{ weights: ", "")
			.replace("], [", "\n\n")
			.replace(" } }", "")
	);
}
