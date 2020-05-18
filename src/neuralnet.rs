extern crate rand;
extern crate rulinalg;

use rand::Rng;
use rulinalg::matrix::{Matrix, BaseMatrixMut, BaseMatrix};

/* Math ------------------------------------------ */

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

fn dsigmoid(y: f64) -> f64 {
    return y * (1.0 - y);
}

/* Data ------------------------------------------ */

#[derive(Debug)]
struct Data {
    inputs: Matrix<f64>,
    targets: Matrix<f64>
}

/* Neural Network -------------------------------- */

struct NeuralNetwork {
    weights_ih: Matrix<f64>,
    weights_ho: Matrix<f64>,
    bias_h: Matrix<f64>,
    bias_o: Matrix<f64>,
    learning_rate: f64
}

impl NeuralNetwork {
    fn new(nb_inputs: usize, nb_hidden: usize, nb_outputs: usize) -> NeuralNetwork {
        NeuralNetwork {

            weights_ih: Matrix::new(nb_hidden, nb_inputs, (0..(nb_hidden * nb_inputs)).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            weights_ho: Matrix::new(nb_outputs, nb_hidden, (0..(nb_outputs * nb_hidden)).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_h: Matrix::new(nb_hidden, 1, (0..nb_hidden).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            bias_o: Matrix::new(nb_outputs, 1, (0..nb_outputs).map(|_| rand::thread_rng().gen_range(-1.0, 1.0)).collect::<Vec<f64>>()),
            learning_rate: 0.1
        }
    }

    fn feedforward(&mut self, inputs: Matrix<f64>) -> Matrix<f64> {
        // Feed forward inputs -> hidden
        let mut hidden: Matrix<f64> = &self.weights_ih * &inputs;
        hidden = &hidden + &self.bias_h;
        hidden = hidden.apply(&sigmoid);
        // Feed forward hidden -> outputs
        let mut outputs: Matrix<f64> = &self.weights_ho * &hidden;
        outputs = &outputs + &self.bias_o;
        outputs = outputs.apply(&sigmoid);
        return outputs;
    }

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Matrix<f64>) { 
        // Feed forward inputs -> hidden
        let mut hidden: Matrix<f64> = &self.weights_ih * inputs;
        hidden = &hidden + &self.bias_h;
        hidden = hidden.apply(&sigmoid);
        // Feed forward hidden -> outputs
        let mut outputs: Matrix<f64> = &self.weights_ho * &hidden;
        outputs = &outputs + &self.bias_o;
        outputs = outputs.apply(&sigmoid);

        // Computing outputs errors
        let output_errors: Matrix<f64> = targets - &outputs;
        // Computing outputs gradient
        let mut outputs_gradients: Matrix<f64> = outputs.apply(&dsigmoid);
        outputs_gradients = outputs_gradients.elemul(&output_errors);
        outputs_gradients = &outputs_gradients * &self.learning_rate;
        // Computing hidden -> outputs deltas
        let hidden_transposed: Matrix<f64> = hidden.transpose();
        let weights_ho_deltas: Matrix<f64> = &outputs_gradients * hidden_transposed;
        // Update hidden -> outputs weights
        self.weights_ho = &self.weights_ho + &weights_ho_deltas;
        // Update outputs bias
        self.bias_o = &self.bias_o + outputs_gradients;

        // Computing hidden errors
        let weights_ho_transposed: Matrix<f64> = self.weights_ho.transpose();
        let hidden_errors: Matrix<f64> = &weights_ho_transposed * &output_errors;
        // Computing hidden gradient
        let mut hidden_gradients: Matrix<f64> = hidden.apply(&dsigmoid);
        hidden_gradients = hidden_gradients.elemul(&hidden_errors);
        hidden_gradients = &hidden_gradients * &self.learning_rate;
        // Computing inputs -> hidden deltas
        let inputs_transposed: Matrix<f64> = inputs.transpose(); 
        let weights_ih_deltas: Matrix<f64> = &hidden_gradients * inputs_transposed;
        // Update inputs -> hidden weights
        self.weights_ih = &self.weights_ih + &weights_ih_deltas;
        // Update outputs bias
        self.bias_h = &self.bias_h + hidden_gradients;
    }
}

/* Functions ------------------------------------- */

// fn main() {
//     let mut neuralnet: NeuralNetwork = NeuralNetwork::new(3, 3, 2);

//     // Exemple :
//     // > There is a training set for color prediction
//     // > The algorithm try to find out wether the color is closer to white or black
//     for _ in 0..50000 {
//         // Here we generate a random color
//         let color: (i32, i32, i32) = (rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256));
//         // Here we compute the target that the network has to guess
//         let targets_tuple: (f64, f64) = if color.0 + color.1 + color.2 > 300 { (1.0, 0.0) } else { (0.0, 1.0) };

//         // Here we format the color to fit in the neural network
//         let inputs: Matrix<f64> = Matrix::new(3, 1, vec!(color.0 as f64 / 255.0, color.1 as f64 / 255.0, color.2 as f64 / 255.0));
//         // Here we format the target to fit in the neural network
//         let targets: Matrix<f64> = Matrix::new(2, 1, vec!(targets_tuple.0, targets_tuple.1));
//         // Training with the given datas
//         &neuralnet.train(&inputs, &targets);
//     }

//     // Testing result, cmp the answer and the guess
//     for _ in 0..10 {
//         let color: (i32, i32, i32) = (rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256), rand::thread_rng().gen_range(0, 256));
//         println!("color: {:?}", color);
//         let white_answer: f64 = if color.0 + color.1 + color.2 > 300 { 0.0 } else { 1.0 };
//         let black_answer: f64 = if color.0 + color.1 + color.2 > 300 { 1.0 } else { 0.0 };
//         println!("calculated answer    : {}", if color.0 + color.1 + color.2 > 300 { String::from("black") } else { String::from("white") });
//         let result: Matrix<f64> = neuralnet.feedforward(Matrix::new(3, 1, vec!(color.0 as f64 / 255.0, color.1 as f64 / 255.0, color.2 as f64 / 255.0)));
//         println!("neural network guess : {} | fitness: {}%\n", if result.data()[0] > result.data()[1] { String::from("black") } else { String::from("white") }, 100 - (((result.data()[0] * white_answer + result.data()[1] * black_answer) / 2.0) * 100.0) as i32);
//     }
// }