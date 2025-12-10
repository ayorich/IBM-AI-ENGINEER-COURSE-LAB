"""
Document Classification System
A production-ready text classification application using PyTorch.

Usage:
    python document_classifier.py train    # Train the model
    python document_classifier.py predict "Your text here"  # Classify text
    python document_classifier.py serve    # Start API server
"""

import os
import sys
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Configuration ==============
class Config:
    """Configuration class for the classifier."""
    # Model parameters
    EMBED_DIM = 64
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.1
    
    # Paths
    MODEL_PATH = "model/document_classifier.pth"
    VOCAB_PATH = "model/vocab.pt"
    CONFIG_PATH = "model/config.json"
    
    # Labels
    LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    NUM_CLASSES = 4


# ============== Model Definition ==============
class TextClassificationModel(nn.Module):
    """Neural network for text classification using EmbeddingBag."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with uniform distribution."""
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# ============== Document Classifier ==============
class DocumentClassifier:
    """Production-ready document classification system."""
    
    def __init__(self, model_dir: str = "model"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None
        self.model = None
        self.config = Config()
        
        logger.info(f"Using device: {self.device}")
    
    def _yield_tokens(self, data_iter):
        """Generator for tokenized text."""
        for _, text in data_iter:
            yield self.tokenizer(text.lower())
    
    def _text_pipeline(self, text: str) -> List[int]:
        """Convert text to token indices."""
        return self.vocab(self.tokenizer(text.lower()))
    
    def _label_pipeline(self, label: int) -> int:
        """Convert label (1-4) to zero-indexed (0-3)."""
        return int(label) - 1
    
    def _collate_batch(self, batch):
        """Custom collate function for DataLoader."""
        label_list, text_list, offsets = [], [], [0]
        
        for _label, _text in batch:
            label_list.append(self._label_pipeline(_label))
            processed_text = torch.tensor(
                self._text_pipeline(_text), 
                dtype=torch.int64
            )
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        
        return (
            label_list.to(self.device),
            text_list.to(self.device),
            offsets.to(self.device)
        )
    
    def build_vocab(self):
        """Build vocabulary from AG_NEWS training data."""
        logger.info("Building vocabulary...")
        train_iter = AG_NEWS(split="train")
        
        self.vocab = build_vocab_from_iterator(
            self._yield_tokens(train_iter),
            specials=["<unk>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def prepare_data(self):
        """Prepare training, validation, and test datasets."""
        logger.info("Preparing datasets...")
        
        # Load datasets
        train_iter, test_iter = AG_NEWS()
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        
        # Split training data
        num_train = int(len(train_dataset) * 0.95)
        train_dataset, valid_dataset = random_split(
            train_dataset,
            [num_train, len(train_dataset) - num_train]
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_batch
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=self._collate_batch
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=self._collate_batch
        )
        
        logger.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, valid_loader, test_loader
    
    def build_model(self):
        """Build the classification model."""
        self.model = TextClassificationModel(
            vocab_size=len(self.vocab),
            embed_dim=self.config.EMBED_DIM,
            num_class=self.config.NUM_CLASSES
        ).to(self.device)
        
        logger.info(f"Model built: {self.model}")
        return self.model
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model accuracy on a dataset."""
        self.model.eval()
        total_acc, total_count = 0, 0
        
        with torch.no_grad():
            for label, text, offsets in dataloader:
                predicted = self.model(text, offsets)
                total_acc += (predicted.argmax(1) == label).sum().item()
                total_count += label.size(0)
        
        return total_acc / total_count
    
    def train(self, epochs: Optional[int] = None):
        """Train the model."""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        # Build vocab and prepare data
        self.build_vocab()
        train_loader, valid_loader, test_loader = self.prepare_data()
        self.build_model()
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        
        best_accuracy = 0.0
        loss_history = []
        accuracy_history = []
        
        # Training loop
        logger.info("Starting training...")
        for epoch in tqdm(range(1, epochs + 1), desc="Training"):
            self.model.train()
            epoch_loss = 0.0
            
            for label, text, offsets in train_loader:
                optimizer.zero_grad()
                predicted = self.model(text, offsets)
                loss = criterion(predicted, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                epoch_loss += loss.item()
            
            # Evaluate
            val_accuracy = self._evaluate(valid_loader)
            loss_history.append(epoch_loss)
            accuracy_history.append(val_accuracy)
            
            logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Val Acc={val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model()
        
        # Final evaluation
        test_accuracy = self._evaluate(test_loader)
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")
        
        return {
            "loss_history": loss_history,
            "accuracy_history": accuracy_history,
            "test_accuracy": test_accuracy
        }
    
    def save_model(self):
        """Save model, vocab, and config to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), self.model_dir / "classifier.pth")
        
        # Save vocabulary
        torch.save(self.vocab, self.model_dir / "vocab.pt")
        
        # Save config
        config_data = {
            "vocab_size": len(self.vocab),
            "embed_dim": self.config.EMBED_DIM,
            "num_classes": self.config.NUM_CLASSES,
            "labels": self.config.LABELS
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model saved to {self.model_dir}")
    
    def load_model(self):
        """Load model, vocab, and config from disk."""
        # Load config
        with open(self.model_dir / "config.json", "r") as f:
            config_data = json.load(f)
        
        # Load vocabulary
        self.vocab = torch.load(self.model_dir / "vocab.pt")
        
        # Build and load model
        self.model = TextClassificationModel(
            vocab_size=config_data["vocab_size"],
            embed_dim=config_data["embed_dim"],
            num_class=config_data["num_classes"]
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(self.model_dir / "classifier.pth", map_location=self.device)
        )
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text document.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predicted label and confidence scores
        """
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        with torch.no_grad():
            text_tensor = torch.tensor(self._text_pipeline(text), dtype=torch.int64)
            offset_tensor = torch.tensor([0])
            
            # Move to device
            text_tensor = text_tensor.to(self.device)
            offset_tensor = offset_tensor.to(self.device)
            
            # Get predictions
            output = self.model(text_tensor, offset_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            predicted_idx = output.argmax(1).item()
            predicted_label = self.config.LABELS[predicted_idx]
            confidence = probabilities[0][predicted_idx].item()
            
            # Get all class probabilities
            class_probs = {
                self.config.LABELS[i]: probabilities[0][i].item()
                for i in range(self.config.NUM_CLASSES)
            }
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_probabilities": class_probs
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple text documents."""
        return [self.predict(text) for text in texts]


# ============== Flask API Server ==============
def create_api_app(classifier: DocumentClassifier):
    """Create Flask API application."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error("Flask not installed. Run: pip install flask")
        return None
    
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})
    
    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()
        
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        result = classifier.predict(data["text"])
        return jsonify(result)
    
    @app.route("/predict_batch", methods=["POST"])
    def predict_batch():
        data = request.get_json()
        
        if "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400
        
        results = classifier.predict_batch(data["texts"])
        return jsonify({"results": results})
    
    return app


# ============== CLI Interface ==============
def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    classifier = DocumentClassifier()
    
    if command == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = classifier.train(epochs=epochs)
        print(f"\nTraining complete! Test accuracy: {results['test_accuracy']:.2%}")
    
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python document_classifier.py predict \"Your text here\"")
            sys.exit(1)
        
        text = " ".join(sys.argv[2:])
        result = classifier.predict(text)
        
        print("\n" + "=" * 50)
        print("DOCUMENT CLASSIFICATION RESULT")
        print("=" * 50)
        print(f"Text: {result['text']}")
        print(f"\nPredicted Category: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll Probabilities:")
        for label, prob in result['class_probabilities'].items():
            print(f"  {label}: {prob:.2%}")
    
    elif command == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        classifier.load_model()
        app = create_api_app(classifier)
        
        if app:
            print(f"\nStarting API server on http://localhost:{port}")
            print("Endpoints:")
            print("  POST /predict - Classify single document")
            print("  POST /predict_batch - Classify multiple documents")
            print("  GET /health - Health check")
            app.run(host="0.0.0.0", port=port)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict, serve")
        sys.exit(1)


if __name__ == "__main__":
    main()