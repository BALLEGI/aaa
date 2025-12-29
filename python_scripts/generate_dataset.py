import json
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
import os
import ai_utils # Importation de notre module utils

# --- CONFIGURATION ---
KAFKA_SERVER = os.environ.get('KAFKA_SERVER', 'kafka:29092')
NETWORK_TOPIC = os.environ.get('NETWORK_TOPIC', 'network_logs')
CSV_FILE = '/models/training_data.csv'
DATA_COUNT = 3000 # Nombre de logs à ingérer

def generate_data():
    print(f"Génération du dataset ({DATA_COUNT} messages)...")
    consumer = KafkaConsumer(
        NETWORK_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    features_list = []
    count = 0

    for message in consumer:
        data = message.value
        if data.get('_path') == 'conn':
            try:
                vector = ai_utils.extract_features(data)
                features_list.append(vector)
                count += 1
                if count >= DATA_COUNT:
                    break
            except Exception as e:
                pass # Ignorer lignes corrompues

    if not features_list:
        print("ERREUR: Aucune donnée Zeek reçue. Kafka est-il vide ?")
        return

    # Création du DataFrame
    columns = ['port', 'proto', 'duration', 'orig_bytes', 'resp_bytes', 'state']
    df = pd.DataFrame(features_list, columns=columns)
    df.to_csv(CSV_FILE, index=False)
    print(f"Dataset OK: {len(df)} lignes dans {CSV_FILE}")
    return df

if __name__ == "__main__":
    generate_data()
