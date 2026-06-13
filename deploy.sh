#!/usr/bin/env bash
# Production launch script for Quotex Signal Pro (100k users)
# Usage: ./deploy.sh [dev|prod]

set -euo pipefail

MODE="${1:-dev}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Quotex Signal Pro — Deploy ($MODE)"
echo "============================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose not found."
    exit 1
fi

# Check .env file
if [ ! -f "$PROJECT_DIR/backend/.env" ]; then
    echo "ERROR: backend/.env not found. Create it with:"
    echo "  QUOTEX_EMAIL=your@email.com"
    echo "  QUOTEX_PASSWORD=your_password"
    exit 1
fi

cd "$PROJECT_DIR"

case "$MODE" in
    dev)
        echo ""
        echo "[DEV MODE] Starting single worker + Redis..."
        echo ""
        docker-compose up --build redis api
        ;;

    prod)
        echo ""
        echo "[PROD MODE] Starting full stack (4 workers + Redis + Nginx)..."
        echo ""

        # Build and start
        docker-compose up -d --build --remove-orphans

        echo ""
        echo "============================================"
        echo "  Services started:"
        echo "  - Nginx:      http://localhost:80"
        echo "  - API (x4):   internal:8000"
        echo "  - Redis:      localhost:6379"
        echo "============================================"
        echo ""
        echo "  Workers:  $(docker-compose ps api | grep -c 'Up' || echo 0)"
        echo "  Redis:    $(docker-compose ps redis | grep -c 'Up' || echo 0)"
        echo "  Nginx:    $(docker-compose ps nginx | grep -c 'Up' || echo 0)"
        echo ""

        # Health check
        sleep 5
        echo "  Health check:"
        curl -s http://localhost/api/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (waiting for services to start...)"
        ;;

    scale)
        WORKERS="${2:-8}"
        echo ""
        echo "[SCALE] Scaling API to $WORKERS workers..."
        docker-compose up -d --scale api="$WORKERS" --no-recreate
        echo "  Scaled to $WORKERS workers"
        ;;

    stop)
        echo ""
        echo "[STOP] Stopping all services..."
        docker-compose down
        echo "  All services stopped."
        ;;

    logs)
        docker-compose logs -f --tail=100
        ;;

    status)
        echo ""
        echo "[STATUS]"
        docker-compose ps
        echo ""
        if curl -s http://localhost/api/health > /dev/null 2>&1; then
            echo "API Health:"
            curl -s http://localhost/api/health | python3 -m json.tool
        else
            echo "API: not responding"
        fi
        ;;

    *)
        echo "Usage: $0 [dev|prod|scale|stop|logs|status]"
        echo ""
        echo "  dev    - Start single worker + Redis (development)"
        echo "  prod   - Start full stack (4 workers + Redis + Nginx)"
        echo "  scale  - Scale API workers: $0 scale <count>"
        echo "  stop   - Stop all services"
        echo "  logs   - Tail logs"
        echo "  status - Show service status + health"
        exit 1
        ;;
esac
