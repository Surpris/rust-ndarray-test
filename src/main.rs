extern crate rust_ndarray_test;

use ndarray::{array, Array1};
use std::time::Instant;
use rust_ndarray_test::common::ndarray_init::*;

macro_rules! measure {
  ( $x:expr) => {
    {
      let start = Instant::now();
      let result = $x;
      let end = start.elapsed();
      let elapsed: f64 = end.as_secs() as f64 + end.subsec_micros() as f64 * 1E-6;
      println!("elapsed time: {} sec.", elapsed);
      result
    }
  };
}

fn test_create_array(){
    // println!("test_create_array");
    // let start = Instant::now();
    let  _: Array1<f64> = array![0., 1. ,2.];
    // let end = start.elapsed();
    // let elapsed: f64 = end.as_secs() as f64 + end.subsec_micros() as f64 * 1E-6;
    // println!("{}", a);
    // println!("elapsed time: {} sec.", elapsed);
}

fn test_create_randomized_ndarray() {
    println!("test_randomized_ndarray");
    let n: usize = 10;
    let mu: f64 = 1.0;
    let sigma: f64 = 0.5;

    let _: Array1<f64> = initialize_randomized_ndarray(DistributionEnum::Normal, n, &[mu, sigma]);
    // println!("{}", a);
}

fn main() {
    measure!(test_create_array());
    measure!(test_create_randomized_ndarray());
    println!("Hello, world!");
}
