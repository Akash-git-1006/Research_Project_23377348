"""
Kubernetes Operator for Adaptive Model Deployment with Epsilon-Greedy Routing

This operator:
1. Watches for ModelDeploymentGroup custom resources
2. Collects metrics from Prometheus
3. Implements epsilon-greedy algorithm for model selection
4. Updates traffic routing weights based on performance
"""
import kopf
import kubernetes
import logging
import time
import random
import requests
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Epsilon-greedy configuration
EPSILON = 0.2  # 20% exploration, 80% exploitation
PROMETHEUS_URL = "http://prometheus-server.prometheus.svc.cluster.local"
METRICS_COLLECTION_INTERVAL = 30  # seconds

# Model performance tracker
model_stats = defaultdict(lambda: {
    'requests': 0,
    'errors': 0,
    'total_latency': 0.0,
    'avg_latency': 0.0,
    'error_rate': 0.0,
    'reward': 0.0  # Combined metric for epsilon-greedy
})


class EpsilonGreedyRouter:
    """
    Implements epsilon-greedy multi-armed bandit algorithm for model selection
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon
        self.model_rewards = defaultdict(float)
        self.model_counts = defaultdict(int)

    def calculate_reward(self, latency, error_rate):
        """
        Calculate reward for a model based on latency and error rate

        Higher reward = better performance
        Reward = 1 / (latency * (1 + error_rate * 10))
        """
        if latency <= 0:
            latency = 0.001  # Avoid division by zero

        # Penalize errors heavily
        error_penalty = 1 + (error_rate * 10)

        # Lower latency = higher reward
        reward = 1.0 / (latency * error_penalty)

        return reward

    def select_model(self, available_models):
        """
        Select a model using epsilon-greedy strategy

        With probability epsilon: explore (random selection)
        With probability (1-epsilon): exploit (best model)
        """
        if not available_models:
            return None

        # Exploration: random selection
        if random.random() < self.epsilon:
            selected = random.choice(available_models)
            logger.info(f"EXPLORATION: Randomly selected {selected}")
            return selected

        # Exploitation: select best performing model
        best_model = None
        best_reward = float('-inf')

        for model in available_models:
            reward = self.model_rewards.get(model, 0.0)
            if reward > best_reward:
                best_reward = reward
                best_model = model

        # If no model has been evaluated yet, select randomly
        if best_model is None:
            best_model = random.choice(available_models)

        logger.info(f"EXPLOITATION: Selected best model {best_model} (reward: {best_reward:.4f})")
        return best_model

    def update_model(self, model_name, reward):
        """Update model statistics"""
        self.model_counts[model_name] += 1
        n = self.model_counts[model_name]

        # Running average of rewards
        old_reward = self.model_rewards[model_name]
        self.model_rewards[model_name] = old_reward + (reward - old_reward) / n

        logger.info(f"Updated {model_name}: reward={self.model_rewards[model_name]:.4f}, count={n}")

    def get_traffic_weights(self, available_models):
        """
        Calculate traffic distribution weights based on model performance

        Returns a dictionary of model -> weight (0-1, sum=1)
        """
        if not available_models:
            return {}

        weights = {}
        total_reward = sum(self.model_rewards.get(m, 0.1) for m in available_models)

        if total_reward == 0:
            # Equal weights if no data
            weight = 1.0 / len(available_models)
            for model in available_models:
                weights[model] = weight
        else:
            # Proportional to rewards, with epsilon for exploration
            for model in available_models:
                reward = self.model_rewards.get(model, 0.1)
                # Softmax-like distribution with exploration bonus
                weights[model] = (reward / total_reward) * (1 - self.epsilon) + (self.epsilon / len(available_models))

        return weights


# Initialize router
router = EpsilonGreedyRouter(epsilon=EPSILON)


def query_prometheus(query):
    """Query Prometheus for metrics"""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={'query': query},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return data['data']['result']
        logger.error(f"Prometheus query failed: {response.text}")
    except Exception as e:
        logger.error(f"Error querying Prometheus: {e}")
    return []


def collect_model_metrics():
    """Collect metrics from Prometheus for all models"""
    logger.info("Collecting metrics from Prometheus...")

    models = ['logistic-regression', 'random-forest', 'svm']

    for model_name in models:
        # Query success requests
        success_query = f'model_requests_total{{model_name="{model_name}",status="success"}}'
        success_results = query_prometheus(success_query)

        # Query error requests
        error_query = f'model_requests_total{{model_name="{model_name}",status="error"}}'
        error_results = query_prometheus(error_query)

        # Query latency
        latency_query = f'rate(model_request_duration_seconds_sum{{model_name="{model_name}"}}[5m]) / rate(model_request_duration_seconds_count{{model_name="{model_name}"}}[5m])'
        latency_results = query_prometheus(latency_query)

        # Extract values
        success_count = float(success_results[0]['value'][1]) if success_results else 0
        error_count = float(error_results[0]['value'][1]) if error_results else 0
        avg_latency = float(latency_results[0]['value'][1]) if latency_results else 0.1

        total_requests = success_count + error_count
        error_rate = error_count / total_requests if total_requests > 0 else 0.0

        # Update stats
        model_stats[model_name]['requests'] = total_requests
        model_stats[model_name]['errors'] = error_count
        model_stats[model_name]['avg_latency'] = avg_latency
        model_stats[model_name]['error_rate'] = error_rate

        # Calculate reward
        reward = router.calculate_reward(avg_latency if avg_latency > 0 else 0.1, error_rate)
        model_stats[model_name]['reward'] = reward

        # Update router
        router.update_model(model_name, reward)

        logger.info(f"{model_name}: requests={total_requests}, errors={error_count}, "
                    f"latency={avg_latency:.4f}s, error_rate={error_rate:.2%}, reward={reward:.4f}")

    # Calculate and log traffic weights
    weights = router.get_traffic_weights(models)
    logger.info(f"Traffic weights: {json.dumps({k: f'{v:.2%}' for k, v in weights.items()})}")

    return weights


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    """Configure operator settings"""
    settings.posting.enabled = False  # Disable event posting
    logger.info("Adaptive Router Operator started!")


@kopf.timer('deployments', interval=METRICS_COLLECTION_INTERVAL, labels={'app': 'ml-model'})
def monitor_models(spec, name, namespace, **kwargs):
    """
    Periodic monitoring of model deployments

    This function runs every METRICS_COLLECTION_INTERVAL seconds
    """
    logger.info(f"Monitoring model deployment: {name}")

    # Collect metrics
    weights = collect_model_metrics()

    # Log current routing decisions
    models = list(weights.keys())
    selected = router.select_model(models)
    logger.info(f"Current best model for routing: {selected}")


@kopf.on.create('services', labels={'app': 'ml-model'})
def model_service_created(spec, name, namespace, **kwargs):
    """Handle new model service creation"""
    logger.info(f"New model service detected: {name} in {namespace}")
    logger.info("Will start monitoring in next interval...")


if __name__ == '__main__':
    logger.info("Starting Adaptive Router Operator...")
    logger.info(f"Epsilon: {EPSILON}")
    logger.info(f"Prometheus URL: {PROMETHEUS_URL}")
    logger.info(f"Metrics collection interval: {METRICS_COLLECTION_INTERVAL}s")

    # Run operator
    kopf.run()
