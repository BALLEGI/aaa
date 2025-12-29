import json, os, time
from kafka import KafkaConsumer
from neo4j import GraphDatabase

KAFKA_SERVER = os.environ.get('KAFKA_SERVER', 'kafka:29092')
NETWORK_TOPIC = os.environ.get('NETWORK_TOPIC', 'network_logs')
WAZUH_TOPIC = os.environ.get('WAZUH_TOPIC', 'wazuh_alerts')
AI_ALERT_TOPIC = 'ai_anomaly_alerts' # IMPORTANT : Le topic IA
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_AUTH = (os.environ.get('NEO4J_USER','neo4j'), os.environ.get('NEO4J_PASSWORD','password_aion'))

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# --- FONCTIONS GRAPHES ---

def add_connection(tx, src, dst, port, proto):
    query = """
    MERGE (h1:Host {ip: $src}) MERGE (h2:Host {ip: $dst})
    MERGE (h1)-[r:CONNECTED_TO {port:$port, proto:$proto}]->(h2)
    SET r.count = coalesce(r.count,0) + 1
    """
    tx.run(query, src=src, dst=dst, port=port, proto=proto)

def add_wazuh_alert(tx, agent_name, level, desc):
    query = """
    MERGE (a:Agent {name: $name})
    MERGE (al:Alert {desc: $desc, type: 'WAZUH', time: datetime()})
    MERGE (a)-[:GENERATED]->(al) SET al.level = $level
    """
    tx.run(query, name=agent_name, level=level, desc=desc)

def add_ai_alert(tx, src_ip, score, dst_port, desc):
    # NOUVEAU : Intégration des alertes IA
    query = """
    MERGE (h:Host {ip: $src_ip})
    CREATE (al:Alert {
        desc: $desc, 
        type: 'AI_VAE', 
        score: $score, 
        time: datetime(),
        dst_port: $dst_port
    })
    CREATE (h)-[:GENERATED]->(al)
    """
    tx.run(query, src_ip=src_ip, score=score, desc=desc, dst_port=dst_port)

# --- TRAITEMENT ---

def process_msg(msg):
    if msg.topic == NETWORK_TOPIC:
        d = msg.value
        if d.get('_path') == 'conn':
            src, dst, p, proto = d.get('id.orig_h'), d.get('id.resp_h'), d.get('id.resp_p'), d.get('proto')
            if src and dst:
                with driver.session() as s: s.execute_write(add_connection, src, dst, p, proto)
    
    elif msg.topic == WAZUH_TOPIC:
        d = msg.value
        level = d.get('rule', {}).get('level')
        if level and level >= 3:
            agent = d.get('agent', {}).get('name')
            desc = d.get('rule', {}).get('description')
            if agent:
                with driver.session() as s: s.execute_write(add_wazuh_alert, agent, level, desc)

    elif msg.topic == AI_ALERT_TOPIC:
        # TRAITEMENT IA
        d = msg.value
        src_ip = d.get('src_ip')
        score = d.get('score')
        dst_port = d.get('dst_port')
        desc = d.get('description')
        
        if src_ip:
            with driver.session() as s: s.execute_write(add_ai_alert, src_ip, score, dst_port, desc)

# --- MAIN ---

consumer = KafkaConsumer(
    NETWORK_TOPIC, 
    WAZUH_TOPIC, 
    AI_ALERT_TOPIC, # Écoute de l'IA
    bootstrap_servers=[KAFKA_SERVER], 
    auto_offset_reset='earliest', 
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Unified Graph Builder Running (Zeek + Wazuh + AI)...")
for msg in consumer: process_msg(msg)
