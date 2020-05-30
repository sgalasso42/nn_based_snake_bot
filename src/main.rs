extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;
extern crate rand;
extern crate rulinalg;

mod neuralnet;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::*;
use piston::input::*;
use piston::window::WindowSettings;
use rand::Rng;
use rulinalg::matrix::{Matrix, BaseMatrix};
use std::cmp::Ordering;

use neuralnet::NeuralNetwork;

// RGBA
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

// Cell Type
const EMPTY: i32 = 0;
const FOOD: i32 = 1;
const SNAKE: i32 = 2;
const HEAD: i32 = 3;

/* Direction ------------------------------------------------------ */

#[derive(Clone, PartialEq)]
enum Direction {
    N, S, W, E, None
}

/* Display -------------------------------------------------------- */

struct Display {
    gl: GlGraphics,
}

impl Display {
    fn new(opengl: OpenGL) -> Display {
        return Display {
            gl: GlGraphics::new(opengl)
        }
    }

    fn clear(&mut self, args: &RenderArgs, x: f64, y: f64, width: f64, height: f64, game_over: bool) {
        let border_rect = graphics::rectangle::rectangle_by_corners(x, y, x + width, y + height);
        let fill_rect = graphics::rectangle::rectangle_by_corners(x + 1.0, y + 1.0, x + width - 1.0, y + height - 1.0);

        self.gl.draw(args.viewport(), |c, gl| {
            graphics::rectangle(if game_over { RED } else { WHITE }, border_rect, c.transform, gl);
            graphics::rectangle(BLACK, fill_rect, c.transform, gl);
        });
    }

    // fn render_point(&mut self, args: &RenderArgs, x: f64, y: f64, size: f64, color: [f32; 4]) {
    //     let square = graphics::rectangle::square(x, y, size);
    
    //     self.gl.draw(args.viewport(), |c, gl| {
    //         graphics::ellipse(color, square, c.transform, gl);
    //     });
    // }

    fn render_rectangle(&mut self, args: &RenderArgs, x: f64, y: f64, width: f64, height: f64, color: [f32; 4]) {
        let rectangle = graphics::rectangle::rectangle_by_corners(x, y, x + width, y + height);
    
        self.gl.draw(args.viewport(), |c, gl| {
            graphics::rectangle(color, rectangle, c.transform, gl);
        });
    }

    fn render_line(&mut self, args: &RenderArgs) {
        self.gl.draw(args.viewport(), |c, gl| {
            graphics::line(BLACK, 0.5, [0.0, 0.0, 500.0, 500.0], c.transform, gl);
        });
    }
}

/* Vec2 ----------------------------------------------------------- */

#[derive(Debug, Clone)]
struct Vec2 {
    x: i32,
    y: i32
}

impl Vec2 {
    fn new() -> Vec2 {
        Vec2 {
            x: 0,
            y: 0
        }
    }
}

/* Snake ---------------------------------------------------------- */

#[derive(Clone)]
struct Snake {
    brain: NeuralNetwork,
    body: Vec<Vec2>,
    dir: Direction
}

impl Snake {
    fn new(cell_nb: usize, brain: NeuralNetwork) -> Snake {
        Snake {
            brain: brain,
            body: (0..3).map(|i| Vec2 {x: cell_nb as i32 / 2 + i, y: cell_nb as i32 / 2}).collect(),
            dir: Direction::None
        }
    }

    fn grow(&mut self) {
        let last = &self.body[self.body.len() - 1];
        self.body.push(Vec2 {x: last.x, y: last.y});
    }

    fn move_forward(&mut self, cell_nb: i32) -> bool {
        let new_pos = match self.dir {
            Direction::N => Vec2 { x: self.body[0].x, y: self.body[0].y - 1 },
            Direction::S => Vec2 { x: self.body[0].x, y: self.body[0].y + 1 },
            Direction::W => Vec2 { x: self.body[0].x - 1, y: self.body[0].y },
            Direction::E => Vec2 { x: self.body[0].x + 1, y: self.body[0].y },
            Direction::None => Vec2 { x: self.body[0].x, y: self.body[0].y },
        };

        // check wall collision
        if new_pos.x < 0 || new_pos.x >= cell_nb || new_pos.y < 0 || new_pos.y >= cell_nb {
            return false;
        }

        // check tail collission
        for part in self.body.iter() {
            if new_pos.x == part.x && new_pos.y == part.y {
                return false;
            }
        }
        
        let mut index = self.body.len() - 1;
        while index > 0 {
            self.body[index].x = self.body[index - 1].x;
            self.body[index].y = self.body[index - 1].y;
            index -= 1;
        }

        self.body[0] = new_pos;

        return true;
    }

    fn draw(&self, game: &Game, args: &RenderArgs, display: &mut Display) {
        for part in 0..self.body.len() {
            let part_pos = Vec2 {
                x: game.origin.x + self.body[part].x * game.cell_size.x,
                y: game.origin.y + self.body[part].y * game.cell_size.y
            };
            display.render_rectangle(&args, part_pos.x as f64, part_pos.y as f64, game.cell_size.y as f64, game.cell_size.x as f64, WHITE);
        }
    }
}

/* Game ----------------------------------------------------------- */

#[derive(Clone)]
struct Game {
    origin: Vec2,
    height: i32,
    width: i32,
    cell_nb: i32,
    cell_size: Vec2,
    map: Vec<Vec<Vec2>>,
    snake: Snake,
    food: Vec2,
    game_over: bool,
    score: f64,
    time_alive: f64
}

impl Game {
    fn new(origin: Vec2, width: i32, height: i32, cell_nb: i32, snake: Snake) -> Game {
        Game {
            origin: origin,
            height: height,
            width: width,
            cell_nb: cell_nb,
            cell_size: Vec2 {x: width / cell_nb, y: height / cell_nb},
            map: (0..height).map(|y| (0..width).map(|x| Vec2::new()).collect()).collect(),
            snake: snake,
            food: Vec2 {x: rand::thread_rng().gen_range(0, cell_nb), y: rand::thread_rng().gen_range(0, cell_nb)},
            game_over: false,
            score: 0.0,
            time_alive: 0.0
        }
    }
}

/* Functions ------------------------------------------------------ */

// fn get_new_dir_event(key: &ButtonArgs, snake: &Snake) -> Option<Direction> {
//     if key.state == ButtonState::Press {
//         if key.button == Button::Keyboard(Key::Up) && snake.dir != Direction::N && snake.dir != Direction::S {
//             return Some(Direction::N);
//         } else if key.button == Button::Keyboard(Key::Down) && snake.dir != Direction::N && snake.dir != Direction::S {
//             return Some(Direction::S);
//         } else if key.button == Button::Keyboard(Key::Left) && snake.dir != Direction::E && snake.dir != Direction::W {
//             return Some(Direction::W);
//         } else if key.button == Button::Keyboard(Key::Right) && snake.dir != Direction::E && snake.dir != Direction::W {
//             return Some(Direction::E);
//         }
//     }
//     return None;
// }

fn main() {
    let opengl = OpenGL::V3_2;
    let mut window: Window = WindowSettings::new("Snake", [1000, 1000]).graphics_api(opengl).exit_on_esc(true).build().unwrap();
    let mut display = Display::new(opengl);

    let mut test_mode = false;

    let mut generation_nb = 0;

    let max_col = 10;
    let nb_games = 100;
    let cell_nb = 20;

    let mut games: Vec<Game> = (0..nb_games).map(|index| Game::new(
        Vec2 {x: (index % max_col) * (1000 / max_col), y: (index / max_col) * (1000 / max_col)},
        1000 / max_col, 1000 / max_col, cell_nb,
        Snake::new(cell_nb as usize, NeuralNetwork::new(cell_nb as usize * cell_nb as usize, 5, 4))
    )).collect();
    let mut snakes_alive = nb_games;

    let mut events = Events::new(EventSettings::new()).ups(50);
    while let Some(e) = events.next(&mut window) {
        if snakes_alive == 0 {
            // calculate score
            for index in 0..games.len() {
                games[index].score = /*games[index].score * 100.0 + */games[index].time_alive; // maybe to refactor
            }

            // calculate total score
            let mut totalscore = 0.0;
            for game in games.iter() {
                totalscore += game.score;
                // eprintln!("score_raw: {}", game.score);
            }

            eprintln!("gen: {} raw_tot: {}", generation_nb, totalscore);
            
            // calculate fitness
            if totalscore != 0.0 {
                for i in 0..games.len() {
                    games[i].score = games[i].score / totalscore;
                    // eprintln!("score: {}", games[i].score);
                }
            }

            let mut newgames = games.clone();
            for index in 0..games.len() {
                // pick brain
                let mut brain = games[rand::thread_rng().gen_range(0, nb_games) as usize].snake.brain.clone();
                if totalscore > 0.0 {                    
                    let mut r: f64 = rand::thread_rng().gen_range(0.0, 1.0);
                    let mut i = 0;
                    while r > 0.0 {
                        // eprint!("r: {} - games[{}].score: {} ", r, i, games[i].score);
                        r -= games[i].score;
                        // eprintln!("= r: {}", r);
                        i += 1;
                    }
                    let brain = games[i - 1].snake.brain.clone();
                }
                // eprintln!("---");

                // mutate child brain
                brain.mutate(); // rate is in neuralnet.rs / mutate()

                newgames[index].snake = Snake::new(cell_nb as usize, brain);
                newgames[index].time_alive = 0.0;
                newgames[index].score = 0.0;
                newgames[index].game_over = false;
                newgames[index].food = Vec2 {x: rand::thread_rng().gen_range(0, cell_nb), y: rand::thread_rng().gen_range(0, cell_nb)};
            }
            games = newgames;

            snakes_alive = nb_games;
            generation_nb += 1;
        }
    
        if let Some(args) = e.render_args() {
            if test_mode {
                for index in 0..games.len() {
                    display.clear(&args, games[index].origin.x as f64, games[index].origin.y as f64, games[index].width as f64, games[index].height as f64, games[index].game_over);
                    games[index].snake.draw(&games[index], &args, &mut display);
                    let food_pos = Vec2 {
                        x: games[index].origin.x + games[index].food.x * games[index].cell_size.x,
                        y: games[index].origin.y + games[index].food.y * games[index].cell_size.y
                    };
                    display.render_rectangle(&args, food_pos.x as f64, food_pos.y as f64, games[index].cell_size.y as f64, games[index].cell_size.x as f64, WHITE);
                }
            }
        }
    
        if let Some(key) = e.button_args() {
            // if let Some(dir) = get_new_dir_event(&key, &snake) {
            //     snake.dir = dir;
            // }
            if key.state == ButtonState::Press {
                if key.button == Button::Keyboard(Key::Up) {
                    eprintln!("SWITCH MODE");
                    test_mode = if test_mode { false } else { true };
                }
            }
        }
        
        if let Some(u) = e.update_args() {
            if snakes_alive > 0 {
                for index in 0..games.len() {
                    if !games[index].game_over  {

                        let mut inputs: Vec<f64> = (0..(games[index].cell_nb * games[index].cell_nb)).map(|_| EMPTY as f64).collect();
                        inputs[(games[index].food.y * games[index].cell_nb + games[index].food.x) as usize] = FOOD as f64;
                        for part in games[index].snake.body.iter() {
                            inputs[(part.y * games[index].cell_nb + part.x) as usize] = SNAKE as f64;
                        }
                        inputs[(games[index].snake.body[0].y * games[index].cell_nb + games[index].snake.body[0].x) as usize] = HEAD as f64;

                        // for (i, pos) in inputs.iter().enumerate() {
                        //     eprint!("[{}]", pos);
                        //     if i as i32 % games[index].cell_nb == 0 {
                        //         eprint!("\n");
                        //     }
                        // }
                        // eprint!("\n");
            
                        let result: Matrix<f64> = games[index].snake.brain.feedforward(Matrix::new(inputs.len(), 1, inputs));
            
                        let max_index = result.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(index, _)| index);
            
                        let newdir = match max_index {
                            Some(0) => Direction::N,
                            Some(1) => Direction::S,
                            Some(2) => Direction::W,
                            _ => Direction::E,
                        };

                        games[index].snake.dir = newdir;

                        let cell_nb = games[index].cell_nb;
                        games[index].time_alive += 1.0;
                        if games[index].snake.move_forward(cell_nb) == false {
                            games[index].game_over = true;
                            snakes_alive -= 1;
                        }

                        // check infinite loop
                        if games[index].time_alive > 50.0 {
                            games[index].game_over = true;
                            snakes_alive -= 1;
                        }

                        // check food
                        if games[index].snake.body[0].x == games[index].food.x && games[index].snake.body[0].y == games[index].food.y {
                            games[index].snake.grow();
                            games[index].food = Vec2 {x: rand::thread_rng().gen_range(0, games[index].cell_nb), y: rand::thread_rng().gen_range(0, games[index].cell_nb)};
                            games[index].score += 1.0;
                        }
                    }
                }
            }
        }
    }
}