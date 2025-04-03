-- Drop tables if they exist
DROP TABLE IF EXISTS movie_ratings;
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS users;

-- Create users table
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    join_date DATE DEFAULT CURRENT_DATE
);

-- Create movies table
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    release_year INTEGER,
    genre VARCHAR(50),
    director VARCHAR(100),
    runtime_minutes INTEGER
);

-- Create movie_ratings table
CREATE TABLE movie_ratings (
    rating_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    movie_id INTEGER,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    rating_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

-- Insert sample data into users
INSERT INTO users (user_id, username, email) VALUES
(1, 'john_doe', 'john@example.com'),
(2, 'jane_smith', 'jane@example.com'),
(3, 'bob_wilson', 'bob@example.com');

-- Insert sample data into movies
INSERT INTO movies (movie_id, title, release_year, genre, director, runtime_minutes) VALUES
(1, 'The Matrix', 1999, 'Sci-Fi', 'Wachowski Sisters', 136),
(2, 'Inception', 2010, 'Sci-Fi', 'Christopher Nolan', 148),
(3, 'The Shawshank Redemption', 1994, 'Drama', 'Frank Darabont', 142),
(4, 'Pulp Fiction', 1994, 'Crime', 'Quentin Tarantino', 154),
(5, 'The Dark Knight', 2008, 'Action', 'Christopher Nolan', 152);

-- Insert sample data into movie_ratings
INSERT INTO movie_ratings (rating_id, user_id, movie_id, rating, review_text) VALUES
(1, 1, 1, 5, 'A mind-bending masterpiece!'),
(2, 1, 2, 4, 'Complex but fascinating'),
(3, 2, 1, 4, 'Revolutionary for its time'),
(4, 2, 3, 5, 'One of the greatest movies ever made'),
(5, 3, 4, 5, 'A classic that never gets old'),
(6, 3, 5, 4, 'Best superhero movie ever'); 