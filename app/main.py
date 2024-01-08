from ml_model import GenreClassifier
import numpy as np

def get_input_features():
    artist = input("Enter the artist name: ")
    track = input("Enter the track name: ")
    key = input("Enter the key (e.g., C, D#, etc.): ")
    mode = input("Enter the mode (Major or Minor): ")
    tempo = float(input("Enter the tempo: "))
    return key, mode, tempo, artist, track

def main():
    print("Loading and training the genre classification model...")
    classifier = GenreClassifier('app/data/music_genre_training.csv')

    print("Model evaluation on the test set:")
    classifier.evaluate_model()

    print("Enter song features to classify genre or type 'exit' to quit.")
    while True:
        user_input = input("Enter any key to continue or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break

        key, mode, tempo, artist, track = get_input_features()
        try:
            # Predict the genre using the classifier
            genre = classifier.predict((key, mode, tempo))
            print(f"Artist: {artist}, Track: {track}, Predicted genre: {genre[0]}")
        except Exception as e:
            print(f"An error occurred: {e}. Please make sure to enter the correct features.")

if __name__ == "__main__":
    main()
