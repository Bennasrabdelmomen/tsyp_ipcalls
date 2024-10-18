import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
##ena savit el weights wahadhom donc I gotta redo the whole architecture again :p
def build_autoencoder(input_dim, encoding_dim=10):
  autoencoder = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(encoding_dim, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(input_dim, activation='sigmoid')
  ])
  return autoencoder
def preprocess_anomaly(df: pd.DataFrame):
    features = ['Protocol', 'Length']
    X = df[features]
    X_encoded = pd.get_dummies(X, columns=['Protocol'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    input_dim = X_scaled.shape[1]
    return X_scaled, input_dim,df

