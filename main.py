
import argparse
from models import Environment
from visualization import Visualization
from GRN import GRN

def main():
    parser = argparse.ArgumentParser(description='Simulátor závodu buněk k ovum pomocí GRN')
    parser.add_argument(
        '--grn',
        type=str,
        default='network_definitions/stupid_network_example.json',
        help='Cesta k JSON souboru s konfigurací GRN'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='environments/example_environment.json',
        help='Cesta k JSON souboru s konfigurací prostředí'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.1,
        help='Časový krok simulace'
    )
    
    args = parser.parse_args()
    grn = GRN(config_path=args.grn, dt=args.dt)
    env = Environment.from_json(args.env, sperm_grn=grn)
    vis = Visualization(environment=env, dt=args.dt)
    vis.run()

if __name__ == '__main__':
    main()