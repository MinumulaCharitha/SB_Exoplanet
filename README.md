\# SB\_Exoplanet



\*\*XAI-based Exoplanet Habitability Prediction\*\*



This project predicts the habitability of exoplanets using machine learning and explainable AI (XAI) techniques. It is designed to provide insights into which exoplanets could potentially support life.





\*\* Features



\* Predicts habitability scores for exoplanets

\* Uses explainable AI methods for model interpretability

\* Easy-to-run Python application (`app.py`)

\* Includes templates and preprocessed data (excluding cache and database files)







\*\* Project Structure





SB\_Exoplanet/

├── app.py                # Main application

├── templates/            # HTML templates for the app (if any)

├── Requirements.txt      # Python dependencies

├── .gitignore            # Ignores cache, database, and environment files

└── README.md             # Project documentation









\*\* Setup Instructions



1\. \*\*Clone the repository\*\*



git clone https://github.com/MinumulaCharitha/SB\_Exoplanet.git

cd SB\_Exoplanet

```



2\. \*\*Create a virtual environment\*\*



python -m venv venv





3\. \*\*Activate the environment\*\*



\* \*\*Windows:\*\* `venv\\Scripts\\activate`

\* \*\*Linux/Mac:\*\* `source venv/bin/activate`



4\. \*\*Install dependencies\*\*





pip install -r Requirements.txt







\## Running the App





python app.py





\* The app should start and provide an interface (if using templates or web UI)

\* Make sure any `.db` or `.pkl` files are in place if required by the app







\## Notes



\* `.gitignore` ensures that temporary files, caches, and the database are \*\*not tracked\*\* by Git

\* Use Python 3.7+ for compatibility

\* The project uses machine learning models which may require training or pre-existing model files







\## License



This project is for educational and personal use.

Feel free to adapt and expand it for your own research purposes.



