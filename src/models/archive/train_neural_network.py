import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import mlflow
from src.data.pytorch_dataset import MovieLensDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow experiment globally
mlflow.set_experiment("movielens_experiment")

class HybridRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=32):
        super(HybridRecommender, self).__init__()
        
        # ID Embeddings (Main memory)
        # Note: Index 0 is reserved for <UNK>
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        
        # Side Information Embeddings
        self.gender_emb = nn.Embedding(2, 4)       
        self.age_emb = nn.Embedding(10, 4)         
        self.occ_emb = nn.Embedding(25, 8)         
        
        # Calculate input size
        self.input_dim = embedding_dim * 2 + 4 + 4 + 8 + num_genres
        
        # MLP Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, user, movie, gender, age, occupation, genres):
        # Look up embeddings
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        g = self.gender_emb(gender)
        a = self.age_emb(age)
        o = self.occ_emb(occupation)
        
        # Concatenate everything
        x = torch.cat([u, m, g, a, o, genres], dim=1)
        
        # Pass through MLP
        return self.fc_layers(x).squeeze()

def train_model(train_path, valid_path, embedding_dim=32, lr=0.001, epochs=10, batch_size=1024, output_dir="models/neural_network"):
    logger.info("Initializing Neural Network training...")
    
    with mlflow.start_run(run_name="NeuralNetwork"):
        # Log Parameters
        mlflow.log_params({
            "embedding_dim": embedding_dim,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "model_type": "NeuralNetwork"
        })
        
        # 1. Load Data
        train_dataset = MovieLensDataset(train_path)
        valid_dataset = MovieLensDataset(valid_path, user_map=train_dataset.user_map, movie_map=train_dataset.movie_map)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        num_users = train_dataset.get_num_users()
        num_movies = train_dataset.get_num_movies()
        num_genres = train_dataset.get_num_genres()
        
        logger.info(f"Model Inputs -> Users: {num_users}, Movies: {num_movies}, Genres: {num_genres}")
        mlflow.log_param("num_users", num_users)
        
        # 2. Setup Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = HybridRecommender(num_users, num_movies, num_genres, embedding_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 3. Training Loop
        best_valid_loss = float('inf')
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                user = batch['user'].to(device)
                movie = batch['movie'].to(device)
                gender = batch['gender'].to(device)
                age = batch['age'].to(device)
                occ = batch['occupation'].to(device)
                genres = batch['genres'].to(device)
                rating = batch['rating'].to(device)
                
                optimizer.zero_grad()
                outputs = model(user, movie, gender, age, occ, genres)
                loss = criterion(outputs, rating)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    user = batch['user'].to(device)
                    movie = batch['movie'].to(device)
                    gender = batch['gender'].to(device)
                    age = batch['age'].to(device)
                    occ = batch['occupation'].to(device)
                    genres = batch['genres'].to(device)
                    rating = batch['rating'].to(device)
                    
                    outputs = model(user, movie, gender, age, occ, genres)
                    loss = criterion(outputs, rating)
                    total_valid_loss += loss.item()
                    
            avg_valid_loss = total_valid_loss / len(valid_loader)
            rmse = np.sqrt(avg_valid_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Valid RMSE: {rmse:.4f}")
            
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "valid_loss": avg_valid_loss,
                "valid_rmse": rmse
            }, step=epoch)
            
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_model_path = os.path.join(output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                
        logger.info(f"Training complete. Best Valid RMSE: {np.sqrt(best_valid_loss):.4f}")
        mlflow.log_artifact(os.path.join(output_dir, "best_model.pth"))

if __name__ == "__main__":
    if os.path.exists("data/processed/train.csv"):
        train_model(
            train_path="data/processed/train.csv",
            valid_path="data/processed/validate.csv",
            epochs=5
        )
    else:
        logger.warning("Data not found.")
