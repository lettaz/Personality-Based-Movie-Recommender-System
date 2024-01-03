# Movie Recommendation API

## Overview

This API provides personalized movie and genre recommendations based on users' personality traits and previous movie ratings. It utilizes a unique recommendation algorithm that considers user profiles, movie ratings, and genre preferences.

## Features

- **User Registration**: New users can register by inputing their personality trait score and obtain a unique user ID.
- **Personality Data Management**: Retrieve and update personality data for users.
- **Movie Recommendations**: Get personalized movie recommendations for registered users.
- **Genre Recommendations**: Receive genre recommendations based on preferred movies.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.10 or higher
- Pip package manager
- Virtual environment (optional but recommended)

### Installation Steps

1. **Clone the Repository**

    ```bash
    git clone [Your Repository URL]
    cd [Your Project Directory]
    ```

2. **Create and Activate Virtual Environment** (Optional)

    - For Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    - For macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Configuration**

    Modify the `app/config.py` configuration to store environment variables like `MIN_RATINGS_COUNT`.

5. **Start the Server**

    ```bash
    flask run
    ```
    or 

    ```bash
    python run.py
    ```

    The API will be available at `http://127.0.0.1:5000`.

## API Endpoints

1. **Create New User**
   
   - **Endpoint**: `/api/users/new`
   - **Method**: POST
   - **Body**: User personality data in JSON format
   - **Response**: Unique user ID

2. **Get Personality Data**
   
   - **Endpoint**: `/api/personality/<user_id>`
   - **Method**: GET
   - **Response**: User's personality data

3. **Update Personality Data**
   
   - **Endpoint**: `/api/personality`
   - **Method**: POST
   - **Body**: Updated personality data in JSON format
   - **Response**: Confirmation message

4. **Movie Recommendations**
   
   - **Endpoint**: `/api/recommendations/movies`
   - **Method**: POST
   - **Body**: User ID, number of recommendations, and similarity threshold
   - **Response**: List of recommended movies

5. **Genre Recommendations**
   
   - **Endpoint**: `/api/recommendations/genre`
   - **Method**: POST
   - **Body**: User ID, number of genres, and similarity threshold
   - **Response**: List of recommended genres

## API Documentation

For more detailed information about each endpoint, including request and response formats, visit the API documentation: [https://documenter.getpostman.com/view/11604430/2s9YsFDtZ7](POSTMAN API).

## Local Development

For local development and testing, you can use the Flask development server as mentioned in the setup steps. Ensure that you are using the virtual environment to avoid dependency conflicts.

## Contributing

Contributions to improve the API are welcome. Please follow the standard Git workflow - fork the repository, make your changes, and submit a pull request.
