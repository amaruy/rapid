from dataclasses import dataclass
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from gensim.models import Word2Vec
import torch

from .models import WordEmbeddings, EmbeddingConfig

@dataclass
class Word2VecConfig:
    """Configuration for Word2Vec training.
    
    Attributes:
        vector_size: Dimension of word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum word frequency
        workers: Number of worker threads
        sg: Training algorithm (1 for skip-gram, 0 for CBOW)
        epochs: Number of training epochs
    """
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    workers: int = 4
    sg: int = 1
    epochs: int = 10

class EmbeddingTrainer:
    """Handles training of Word2Vec models and creation of embeddings.
    
    Attributes:
        w2v_config: Word2Vec training configuration
        embed_config: Embedding model configuration
        logger: Logger instance
    """
    
    def __init__(
        self,
        w2v_config: Word2VecConfig,
        embed_config: EmbeddingConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.w2v_config = w2v_config
        self.embed_config = embed_config
        self.logger = logger or logging.getLogger(__name__)
        
    def prepare_sentences(
        self,
        data: pd.DataFrame,
        text_column: str,
        group_columns: List[str]
    ) -> List[List[str]]:
        """Prepare sentences for Word2Vec training.
        
        Args:
            data: Input DataFrame
            text_column: Column containing text to embed
            group_columns: Columns to group by for creating sentences
            
        Returns:
            List of tokenized sentences
        """
        data[text_column] = data[text_column].astype(str)
        grouped_data = data.groupby(group_columns)[text_column].apply(list).reset_index()
        sentences = grouped_data[text_column].tolist()
        
        self.logger.info(f"Prepared {len(sentences)} sentences for training")
        return sentences
    
    def train_word2vec(
        self,
        sentences: List[List[str]],
        update_existing: Optional[Word2Vec] = None
    ) -> Word2Vec:
        """Train Word2Vec model on sentences.
        
        Args:
            sentences: List of tokenized sentences
            update_existing: Optional existing model to update
            
        Returns:
            Trained Word2Vec model
        """
        if update_existing is None:
            model = Word2Vec(
                sentences=sentences,
                vector_size=self.w2v_config.vector_size,
                window=self.w2v_config.window,
                min_count=self.w2v_config.min_count,
                workers=self.w2v_config.workers,
                sg=self.w2v_config.sg
            )
            self.logger.info("Initialized new Word2Vec model")
        else:
            model = update_existing
            model.build_vocab(sentences, update=True)
            self.logger.info("Updating existing Word2Vec model")
            
        model.train(
            sentences,
            total_examples=len(sentences),
            epochs=self.w2v_config.epochs
        )
        
        return model
    
    def create_embedding_matrix(
        self,
        model: Word2Vec,
        encoder: 'Encoder'  # type: ignore # Avoiding circular import
    ) -> np.ndarray:
        """Create embedding matrix from Word2Vec model.
        
        Args:
            model: Trained Word2Vec model
            encoder: Label encoder instance
            
        Returns:
            Numpy array containing embedding matrix
        """
        num_labels = len(encoder)
        embedding_matrix = np.zeros((num_labels, self.w2v_config.vector_size))
        
        for label, encoding in encoder.label2code.items():
            if label in model.wv:
                embedding_matrix[encoding] = model.wv[label]
                
        self.logger.info(
            f"Created embedding matrix with shape {embedding_matrix.shape}"
        )
        return embedding_matrix
    
    def train(
        self,
        data: pd.DataFrame,
        encoder: 'Encoder',  # type: ignore
        text_column: str,
        group_columns: List[str],
        save_dir: Optional[Path] = None
    ) -> Tuple[Word2Vec, WordEmbeddings]:
        """Complete training pipeline.
        
        Args:
            data: Input DataFrame
            encoder: Label encoder instance
            text_column: Column containing text to embed
            group_columns: Columns to group by for creating sentences
            save_dir: Optional directory to save models
            
        Returns:
            Tuple of (Word2Vec model, WordEmbeddings model)
        """
        # Prepare and train
        sentences = self.prepare_sentences(data, text_column, group_columns)
        w2v_model = self.train_word2vec(sentences)
        
        # Create embedding matrix and model
        embedding_matrix = self.create_embedding_matrix(w2v_model, encoder)
        embedder = WordEmbeddings(
            num_embeddings=len(encoder),
            embedding_matrix=embedding_matrix,
            config=self.embed_config,
            logger=self.logger
        )
        
        # Save if requested
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            w2v_model.save(str(save_dir / "word2vec.model"))
            embedder.save(save_dir / "embedder.pt")
            self.logger.info(f"Saved models to {save_dir}")
            
        return w2v_model, embedder 