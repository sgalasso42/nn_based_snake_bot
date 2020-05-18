extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;
extern crate rand;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::*;
use piston::input::*;
use piston::window::WindowSettings;
use rand::Rng;

mod neuralnet;

// RGBA
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/* Direction ------------------------------------------------------ */

#[derive(PartialEq)]
enum Direction {
    N, S, W, E
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

    fn clear(&mut self, args: &RenderArgs) {
        self.gl.draw(args.viewport(), |_c, gl| {
            graphics::clear(BLACK, gl);
        });
    }

    fn render_point(&mut self, args: &RenderArgs, x: f64, y: f64, size: f64, color: [f32; 4]) {
        let square = graphics::rectangle::square(x, y, size);
    
        self.gl.draw(args.viewport(), |c, gl| {
            graphics::ellipse(color, square, c.transform, gl);
        });
    }

    fn render_rectangle(&mut self, args: &RenderArgs, x: f64, y: f64, size: f64, color: [f32; 4]) {
        let square = graphics::rectangle::square(x, y, size);
    
        self.gl.draw(args.viewport(), |c, gl| {
            graphics::rectangle(color, square, c.transform, gl);
        });
    }

    fn render_line(&mut self, args: &RenderArgs) {
        self.gl.draw(args.viewport(), |c, gl| {
            graphics::line(BLACK, 0.5, [0.0, 0.0, 500.0, 500.0], c.transform, gl);
        });
    }
}

/* Vec2 ----------------------------------------------------------- */

#[derive(Clone)]
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

struct Snake {
    body: Vec<Vec2>,
    dir: Direction
}

impl Snake {
    fn new() -> Snake {
        Snake {
            body: Vec::new(),
            dir: Direction::W
        }
    }

    fn move_forward(&mut self, game: &mut Game, food: &mut Option<Vec2>) {
        let new_pos = match self.dir {
            Direction::N => Vec2 { x: self.body[0].x, y: self.body[0].y - 1 },
            Direction::S => Vec2 { x: self.body[0].x, y: self.body[0].y + 1 },
            Direction::W => Vec2 { x: self.body[0].x - 1, y: self.body[0].y },
            Direction::E => Vec2 { x: self.body[0].x + 1, y: self.body[0].y },
            _ => { return ; }
        };

        // check wall collision
        if new_pos.x < 0 || new_pos.x >= game.map.len() as i32 || new_pos.y < 0 || new_pos.y >= game.map.len() as i32 {
            game.game_over = true;
            return ;
        }

        // check tail collission
        for part in self.body.iter() {
            if new_pos.x == part.x && new_pos.y == part.y {
                game.game_over = true;
            }
        }

        // check food
        if new_pos.x == food.as_ref().unwrap().x && new_pos.y == food.as_ref().unwrap().y {
            self.body.push(self.body[self.body.len() - 1].clone());
            *food = None;
        }
        
        let mut index = self.body.len() - 1;
        while index > 0 {
            self.body[index].x = self.body[index - 1].x;
            self.body[index].y = self.body[index - 1].y;
            index -= 1;
        }

        self.body[0] = new_pos;
    }

    fn draw(&self, game: &Game, args: &RenderArgs, display: &mut Display) {
        for part in 1..self.body.len() {
            let pos_x = self.body[part].x as f64 * game.cell_size as f64;
            let pos_y = self.body[part].y as f64 * game.cell_size as f64;
            display.render_rectangle(args, pos_x, pos_y, game.cell_size as f64, WHITE);
        }
        let pos_x = self.body[0].x as f64 * game.cell_size as f64;
        let pos_y = self.body[0].y as f64 * game.cell_size as f64;
        display.render_rectangle(args, pos_x, pos_y, game.cell_size as f64, RED);
    }
}

/* Game ----------------------------------------------------------- */

struct Game {
    cell_size: i32,
    map: Vec<Vec<Vec2>>,
    game_over: bool,
}

impl Game {
    fn new(size: i32) -> Game {
        Game {
            cell_size: 20,
            map: (0..size).map(|y| (0..size).map(|x| Vec2::new()).collect()).collect(),
            game_over: false
        }
    }
}

/* Functions ------------------------------------------------------ */

fn get_new_dir_event(key: &ButtonArgs, snake: &Snake) -> Option<Direction> {
    if key.state == ButtonState::Press {
        if key.button == Button::Keyboard(Key::Up) && snake.dir != Direction::N && snake.dir != Direction::S {
            return Some(Direction::N);
        } else if key.button == Button::Keyboard(Key::Down) && snake.dir != Direction::N && snake.dir != Direction::S {
            return Some(Direction::S);
        } else if key.button == Button::Keyboard(Key::Left) && snake.dir != Direction::E && snake.dir != Direction::W {
            return Some(Direction::W);
        } else if key.button == Button::Keyboard(Key::Right) && snake.dir != Direction::E && snake.dir != Direction::W {
            return Some(Direction::E);
        }
    }
    return None;
}

fn main() {
    let size = 25;
    let mut game = Game::new(size);
    let opengl = OpenGL::V3_2;
    let mut window: Window = WindowSettings::new("Snake", [(size * 20) as f64, (size * 20) as f64]).graphics_api(opengl).exit_on_esc(true).build().unwrap();
    let mut display = Display::new(opengl);

    let mut snake: Snake = Snake::new();
    snake.body.push(Vec2 {x: 13, y: 13});
    snake.body.push(Vec2 {x: 14, y: 13});
    snake.body.push(Vec2 {x: 15, y: 13});

    let mut food = Some(Vec2 {x: rand::thread_rng().gen_range(0, 25), y: rand::thread_rng().gen_range(0, 25)});

    let mut events = Events::new(EventSettings::new()).ups(5);
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            display.clear(&args);
            snake.draw(&game, &args, &mut display);
            display.render_rectangle(&args, (food.as_ref().unwrap().x * game.cell_size) as f64, (food.as_ref().unwrap().y * game.cell_size) as f64, game.cell_size as f64, WHITE);
        }
        if let Some(key) = e.button_args() {
            if let Some(dir) = get_new_dir_event(&key, &snake) {
                snake.dir = dir;
            }
        }
        if !game.game_over  {
            if let Some(u) = e.update_args() {
                snake.move_forward(&mut game, &mut food);
            }
        }
        if food.is_none() {
            food = Some(Vec2 {x: rand::thread_rng().gen_range(0, 25), y: rand::thread_rng().gen_range(0, 25)});
        }
    }
}