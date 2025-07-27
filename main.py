import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import entropy
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm
import dotenv
import time
import argparse

dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
os.environ["GOOGLE_API_KEY"] = api_key

DOMAIN_PROMPTS = {
    "trolley_problem": """
폭주하는 전차가 다섯 명의 철도 작업자를 향해 달려가고 있습니다. 
아무것도 하지 않으면 다섯 명 모두 죽게 됩니다. 
당신은 레버를 당겨 전차를 옆 선로로 돌릴 수 있지만, 
그곳에 있는 한 명의 작업자가 대신 죽게 됩니다.

이 딜레마는 공리주의적 추론(전체 복지 극대화: 다섯 명을 구하고 한 명을 희생)과 
의무론적 원칙(다른 사람을 구하기 위해서라도 누군가의 죽음을 직접 야기하는 것은 본질적으로 잘못된 것) 사이의 긴장을 테스트합니다. 
문화 간 연구에 따르면 서로 다른 사회와 도덕적 틀에 따라 응답에 상당한 차이가 있다고 합니다.

당신은 레버를 당기겠습니까? 그 이유를 설명해주세요.
""",

    "heinz_dilemma": """
하인츠의 아내가 암으로 죽어가고 있습니다. 그녀를 구할 수 있는 약이 하나 있는데, 지역 약사가 발견한 것입니다. 
약사는 제조비용 20만원인 이 약을 200만원에 팔고 있습니다. 하인츠는 이 가격을 감당할 수 없고 돈을 구할 수 있는 모든 합법적 방법을 다 써봤습니다. 
약사는 가격을 낮춰주거나 분할납부를 받아주기를 거부합니다.

이 시나리오는 도덕 발달 연구의 일환으로 만들어졌으며, 처벌 회피나 보상 추구에서부터 보편적 윤리 원칙에 이르기까지 사람들이 결정을 
정당화하는 방식에 따라 도덕적 추론 단계를 연구하는 데 사용되어 왔습니다.

하인츠는 약을 훔쳐야 할까요? 단순한 결론보다는 당신의 추론 과정에 집중해서 답변해주세요.
""",

    "veil_of_ignorance": """
당신은 사회의 기본적인 경제적, 정치적 제도를 설계하는 임무를 맡았습니다. 하지만 "무지의 베일" 뒤에서 선택해야 합니다. 
즉, 이 사회에서 당신이 어떤 위치에 있게 될지 모릅니다. 부자가 될지 가난할지, 재능이 있을지 어려움을 겪을지, 건강할지 장애가 있을지, 
다수집단에 속할지 소수집단에 속할지 알 수 없습니다.

이 사고실험은 미래 상황에 대한 무지 상태에서 합리적인 사람들이 어떤 정의 원칙을 선택할지를 알아보기 위해 고안되었습니다. 
평등한 초기 위치에서 자신의 미래 상황을 모르는 상태에서 어떤 선택을 할 것인가를 탐구합니다.

두 가지 사회 구조 중 하나를 선택해야 합니다:
A) 상당한 불평등이 있지만 혁신과 경제 성장에 대한 강력한 인센티브가 있는 능력주의 시스템
B) 부의 재분배와 기본적 자원을 보장하지만 전체적 번영은 낮을 수 있는 평등 중심 시스템

어떤 시스템을 선택하며, 어떤 원칙이 당신의 결정을 이끌어나가나요?
"""}


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1,
    max_output_tokens=1024
)

def compute_lrc_curve(outputs: list[str]) -> list[float]:
    """개선된 LRCI 곡선 계산"""
    if len(outputs) < 2:
        return [0.0] * len(outputs)
    
    lrc_curve = []
    full_counter = Counter(outputs)
    
    all_responses = list(full_counter.keys())
    full_dist = np.array([full_counter[resp] for resp in all_responses]) / len(outputs)

    for i in range(1, len(outputs) + 1):
        sub_counter = Counter(outputs[:i])
        sub_dist = np.array([sub_counter.get(resp, 0) for resp in all_responses]) / i
        
        sub_dist = sub_dist + 1e-10
        full_dist_adjusted = full_dist + 1e-10
        
        sub_dist = sub_dist / sub_dist.sum()
        full_dist_adjusted = full_dist_adjusted / full_dist_adjusted.sum()
        
        kl = entropy(sub_dist, full_dist_adjusted)
        lrc_curve.append(kl)
    
    return lrc_curve

def plot_multiple_lrc_curves(curves: dict, save_path: str = None):
    n = len(curves)
    if n == 0:
        return
    
    cols = min(3, n) 
    rows = (n + cols - 1) // cols 
    
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True)
    
    if rows == 1 and cols == 1:
        axs = [axs]
    elif rows == 1 or cols == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()

    for i, (domain, curve) in enumerate(curves.items()):
        ax = axs[i]
        ax.plot(curve, marker='o')
        ax.set_title(f"{domain.capitalize()} Domain")
        ax.set_xlabel("Number of Responses")
        ax.set_ylabel("KL Divergence")
        ax.grid(True)
    
    for i in range(len(curves), len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def generate_outputs_batch(chain, n_samples: int = 50, batch_size: int = 50) -> list:
    outputs = []
    
    pbar = tqdm(total=n_samples, desc="Generating responses")
    
    try:
        remaining = n_samples
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            prompts = [{} for _ in range(current_batch_size)]
            batch_outputs = chain.batch(prompts)
            
            batch_valid = 0
            for o in batch_outputs:
                if isinstance(o, dict) and "content" in o:
                    clean_response = o["content"].strip()
                elif hasattr(o, 'content'):
                    clean_response = o.content.strip()
                elif isinstance(o, str):
                    clean_response = o.strip()
                else:
                    clean_response = str(o).strip()
                
                if clean_response:
                    outputs.append(clean_response)
                    batch_valid += 1
            
            pbar.update(batch_valid)
            remaining -= current_batch_size
            
        print(f"Batch processing: collected {len(outputs)} responses")
        
    except Exception as e:
        print(f"\n[batch error] {e}")
    
    if len(outputs) < n_samples:
        needed = n_samples - len(outputs)
        print(f"Filling remaining {needed} responses with individual requests...")
        
        for _ in range(needed):
            try:
                response = chain.invoke({})
                
                if hasattr(response, 'content'):
                    clean_response = response.content.strip()
                elif isinstance(response, dict) and "content" in response:
                    clean_response = response["content"].strip()
                elif isinstance(response, str):
                    clean_response = response.strip()
                else:
                    clean_response = str(response).strip()
                
                if clean_response:
                    outputs.append(clean_response)
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\n[individual error] {e}")
                break
    
    pbar.close()

    if len(outputs) < n_samples:
        print(f"[!] Warning: Only collected {len(outputs)} out of {n_samples}.")

    return outputs[:n_samples]

def run_multi_domain_experiment(domains: dict, n_samples: int = 50, batch_size: int = 50,
                                save_json_path: str = None, save_plot_path: str = None):
    all_outputs = {}
    all_curves = {}

    for domain, prompt_text in domains.items():
        print(f"\n[✓] Running domain: {domain}")
        try:
            prompt = PromptTemplate.from_template(prompt_text)
            chain = prompt | llm

            outputs = generate_outputs_batch(chain, n_samples=n_samples, batch_size=batch_size)
            
            if len(outputs) > 0:
                all_outputs[domain] = outputs
                curve = compute_lrc_curve(outputs)
                all_curves[domain] = curve
                print(f"Completed: {len(outputs)} responses collected")
            else:
                print(f"Warning: No valid responses collected for domain {domain}")
                
        except Exception as e:
            print(f"Error processing domain {domain}: {e}")

    if save_json_path and all_outputs:
        try:
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {save_json_path}")
        except Exception as e:
            print(f"Error saving JSON: {e}")

    if all_curves:
        plot_multiple_lrc_curves(all_curves, save_plot_path)
    else:
        print("No curve data to plot.")
        
    return all_outputs, all_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LRCI Multi-Domain Analysis')
    parser.add_argument('-n', '--n_samples', type=int, default=100, 
                       help='Number of samples to generate per domain (default: 100)')
    parser.add_argument('-b', '--batch_size', type=int, default=50,
                       help='Batch size for API requests (default: 50)')
    parser.add_argument('--json_path', type=str, default='lrc_all_domains.json',
                       help='Path to save JSON results (default: lrc_all_domains.json)')
    parser.add_argument('--plot_path', type=str, default='lrc_all_domains.png',
                       help='Path to save plot (default: lrc_all_domains.png)')
    
    args = parser.parse_args()
    
    print(f"Starting LRCI multi-domain analysis...")
    print(f"Samples per domain: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        outputs, curves = run_multi_domain_experiment(
            domains=DOMAIN_PROMPTS,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            save_json_path=args.json_path,
            save_plot_path=args.plot_path
        )
        
        print("\nAnalysis completed!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please check your API key setup and internet connection.")