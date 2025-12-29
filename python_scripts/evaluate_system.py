import sys
from neo4j import GraphDatabase
import os

# --- CONFIGURATION ---
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password_aion')

def print_header(title):
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_section(title):
    print(f"\n>>> {title}")
    print("-" * 80)

def run_report():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"ERREUR CONNEXION NEO4J: {e}")
        print("Vérifiez que le conteneur neo4j est lancé.")
        sys.exit(1)

    print_header(" RAPPORT D'ÉVALUATION AION (SOC + NOC + AI) ")
    print(" Système : Intégration Wazuh / Zeek / VAE / Neo4j")
    print("=" * 80)

    with driver.session() as session:
        
        # 1. COMPTAGE GLOBAL
        print_section("1. STATISTIQUES GLOBALES")
        q_total = "MATCH (a:Alert) RETURN count(a) AS total"
        q_wazuh = "MATCH (:Agent)-[:GENERATED]->(a:Alert {type:'WAZUH'}) RETURN count(a) AS total"
        q_ai = "MATCH (:Host)-[:GENERATED]->(a:Alert {type:'AI_VAE'}) RETURN count(a) AS total"
        
        total_alerts = session.run(q_total).single()['total']
        wazuh_alerts = session.run(q_wazuh).single()['total']
        ai_alerts = session.run(q_ai).single()['total']
        
        print(f"  -> Total des Incidents de Sécurité : {total_alerts}")
        print(f"     - Attaques Connues (Wazuh Rules) : {wazuh_alerts}")
        print(f"     - Anomalies Inconnues (AI VAE)   : {ai_alerts}")
        
        if total_alerts == 0:
            print("\n[AVERTISSEMENT] Aucune alerte trouvée. Avez-vous exécuté le bot d'attaque ?")
            return

        # 2. TOP ATTAQUANTS (Corrélation IP)
        print_section("2. TOP ADRESSES IP MALVEILLANTES (Corrélation)")
        q_attackers = """
            MATCH (h:Host)-[r:CONNECTED_TO]->(target:Host)
            WHERE (h)-[:GENERATED]->(:Alert)
            RETURN h.ip AS attacker_ip, count(r) AS connections
            ORDER BY connections DESC LIMIT 5
        """
        result = session.run(q_attackers)
        found = False
        for record in result:
            found = True
            print(f"  - {record['attacker_ip']} : {record['connections']} connexions suspectes")
        if not found: print("  Aucun attaquant identifié.")

        # 3. TOP ANOMALIES IA (Scores Élevés = Zero-Days Probables)
        print_section("3. TOP ANOMALIES IA (Zéro-Day Potentiels)")
        q_ai = """
            MATCH (h:Host)-[:GENERATED]->(a:Alert {type:'AI_VAE'})
            RETURN h.ip AS src, a.score AS score, a.desc AS desc, a.time AS ts
            ORDER BY score DESC LIMIT 5
        """
        result = session.run(q_ai)
        for record in result:
            score_disp = f"{record['score']:.4f}"
            time_disp = record['ts'].strftime('%H:%M:%S')
            print(f"  - [{time_disp}] Score: {score_disp} | {record['src']}")
            print(f"    -> Desc: {record['desc']}")
        if ai_alerts == 0: print("  Aucune anomalie détectée par l'IA.")

        # 4. ANALYSE DES ALERTES WAUH (Sévérité)
        print_section("4. DISTRIBUTION DES ALERTES WAUH (Sévérité)")
        q_sev = """
            MATCH (:Agent)-[:GENERATED]->(a:Alert {type:'WAZUH'})
            RETURN a.level AS level, count(a) AS cnt
            ORDER BY level DESC
        """
        result = session.run(q_sev)
        for record in result:
            print(f"  - Niveau {record['level']} : {record['cnt']} alertes")

        # 5. ÉTAT DES CIBLES
        print_section("5. ÉTAT DES ACTIFS DE SÉCURITÉ")
        q_agents = "MATCH (a:Agent) RETURN a.name AS name"
        result = session.run(q_agents)
        print("  - Agents Wazuh actifs :")
        for record in result:
            print(f"    - {record['name']}")

    print("\n" + "=" * 80)
    print(" FIN DU RAPPORT D'ÉVALUATION ")
    print("=" * 80)
    
    driver.close()

if __name__ == "__main__":
    run_report()
