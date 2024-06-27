from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os

import random

# --- Configuration ---

# API Keys (replace with your actual keys)
KEYs = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU" ,"AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
KEY = random.choice(KEYs)
genai.configure(api_key=KEY)
# Initialize FastAPI app
app = FastAPI()

P = {
    0 : "Act like expert in this domaine and prospective borrower. Evaluate the creditworthiness of the individual applicant based on the following credit report. Highlight key factors affecting their credit score and risk profile: ",
    1: "Conduct a comprehensive analysis of the liquidity position of the company based on the following financial statement. Include key liquidity ratios and their implications for credit risk assessment: ",
    2: "Evaluate the capital structure of the borrower using the following financial statement. Discuss the impact of leverage on credit risk and solvency: ",
    3: "Analyze the profitability trends of the borrower over the past three years using the following financial statement. Assess how profitability indicators influence credit risk: ",
    4: "Assess the creditworthiness of the individual applicant based on the following credit report. Highlight key factors affecting their credit score and risk profile: ",
    5: "Examine the quality of collateral offered by the borrower for a secured loan application. Evaluate its adequacy in mitigating credit risk and protecting lender interests: ",
    6: "Develop a credit scoring model using historical repayment data from the following dataset. Compare different scoring methodologies and their predictive accuracy: ",
    7: "Forecast credit losses under baseline and adverse scenarios for a portfolio of corporate loans using historical default data and economic forecasts. Present your findings and methodology: ",
    8: "Discuss the role of macroeconomic factors such as GDP growth, interest rates, and inflation in influencing credit risk. Use historical data and economic theories to support your analysis: ",
    9: "Compare and contrast the advantages and disadvantages of using internal credit ratings versus external credit ratings for credit risk assessment. Provide examples from the financial industry: ",
    10: "Evaluate the effectiveness of stress testing in identifying potential credit losses during economic downturns. Conduct a stress test analysis using the following scenario and portfolio data: ",
    11: "Analyze the impact of regulatory changes (e.g., Basel III) on credit risk management practices in financial institutions. Discuss adjustments made to comply with new regulatory requirements: ",
    12: "Propose risk mitigation strategies for managing credit risk in a portfolio of consumer loans. Consider diversification, credit enhancement techniques, and risk transfer mechanisms: ",
    13: "Assess the credit risk associated with a specific industry sector (e.g., healthcare, technology) based on recent financial performance and market trends. Provide recommendations for lenders: ",
    14: "Investigate the relationship between credit risk and market risk in a diversified investment portfolio. Analyze how credit risk impacts overall portfolio returns and risk-adjusted performance: ",
    15: "Perform a sensitivity analysis of key assumptions used in a credit risk model. Evaluate how changes in input parameters (e.g., default probabilities, recovery rates) affect credit risk assessments: ",
    16: "Discuss the ethical considerations in credit risk assessment, particularly in relation to fairness, transparency, and discrimination. Propose guidelines for ethical credit risk management practices: ",
    17: "Analyze the credit risk exposure of a multinational corporation with operations in emerging markets. Assess geopolitical risks, currency volatility, and sovereign risk factors: ",
    18: "Examine the impact of environmental, social, and governance (ESG) factors on credit risk analysis. Discuss how sustainability metrics influence lending decisions and risk pricing: ",
    19: "Evaluate the credit risk associated with structured finance products (e.g., collateralized debt obligations, mortgage-backed securities). Assess risk tranching and credit enhancement strategies: ",
    20: "Provide recommendations for improving the accuracy and reliability of credit risk models in light of recent advancements in machine learning and artificial intelligence. Discuss practical implementation challenges and benefits: "
}


@app.get("/")
def read_root():
    return {"Home": "Sahara Analytics - MVP"}

def get_prompt(number: int, context: str):
    d = P.copy()
    if context:
        d[number] = d[number] + f" \n CONTEXT: {context}"
    return d[number]
def get_response(user_input,prompt):   
    generation_config = {
        "temperature": 0.5,  
        "top_p": 0.95,     
        "top_k": 40,        
        "max_output_tokens": 2048,  
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [prompt]}],
    )

    response = chat_session.send_message(user_input)
    return response.text

# show prompt list
@app.get("/prompts")
def get_prompts():  
    return {"prompts": P}
@app.post("/generate")
def generate(message: str, number: int, context: str):
    prompt = get_prompt(number, context)
    response = get_response(message, prompt)
    return {"response": response}