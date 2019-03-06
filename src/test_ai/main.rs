#![warn(clippy::all)]
#![allow(clippy::range_plus_one)]

mod network;

use rand_xorshift::XorShiftRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use network::NeuralNet;

fn main() {
	let rng = RefCell::new(XorShiftRng::from_rng(rand::thread_rng()).unwrap());
	let get_random = || rng.borrow_mut().gen_range(0., 1.);

	let mut net = NeuralNet::new(&[2, 6, 2], &get_random);

	println!(
		"Initialized new network {} \n\n\n",
		format!("{:?}", net.layers).replace("Relu", "\nRelu").replace("Out", "\nOut").replace("{ weights: ", "")
	);

	let dataset: Vec<_> = (0..10_000)
		.map(|_| loop {
			let (x, y): (f64, f64) = (get_random(), get_random());

			let d = (x * x + y * y).sqrt();

			if !(d > 0.3 && d < 0.7) {
				let expected = if d > 0.3 { vec![1., 0.] } else { vec![0., 1.] };
				break (x, y, expected);
			}
		})
		.collect();

	let error_err = |expected: &[f64], result: &[f64]| {
		let error: Vec<_> = expected.iter().zip(result).map(|(exp, res)| exp - res).collect();
		let err = error.iter().fold(0., |sum, v| sum + v * v);
		(error, err)
	};

	(0..100).for_each(|i| {
		(0..10).for_each(|_| {
			dataset.iter().skip(i * 10).take(10).for_each(|(x, y, expected)| {
				let result = net.run(vec![*x, *y]);
				let (error, _) = error_err(&expected, &result);
				net.learn(error, 0.05, 0.2);
			});
		});
	});

	const TESTSAMPLES: usize = 100;
	let total_err = dataset.iter().rev().take(TESTSAMPLES).fold(0., |total_err, (x, y, expected)| {
		let result = net.run(vec![*x, *y]);
		let (_, err) = error_err(&expected, &result);

		println!("x:{:.*} y:{:.*}, r:{:.*?} e:{:?}, err:{:.*}", 2, x, 2, y, 2, result, expected, 1, err);

		total_err + err
	});

	println!("\n\n.............\nTotal test error: {} %\n.............\n\n", total_err / (TESTSAMPLES as f64));

	println!(
		"Network after training {}",
		format!("{:?}", net.layers).replace("Relu", "\nRelu").replace("Out", "\nOut").replace("{ weights: ", "")
	);
}
