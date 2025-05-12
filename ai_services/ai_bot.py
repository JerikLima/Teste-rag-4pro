import os
import re
from decouple import config
import logging
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from supabase import create_client, Client
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    HF_TOKEN = config('HUGGINGFACEHUB_API_TOKEN', default=None)
    SUPABASE_API_URL = "https://fqroxktcumonykbnzjbn.supabase.co"
    SUPABASE_API_KEY = config("SUPABASE_API_KEY")
    if not SUPABASE_API_URL or not SUPABASE_API_KEY:
        raise ValueError("SUPABASE_API_URL e SUPABASE_API_KEY devem ser definidos.")
    logging.info(f"SUPABASE_API_URL = {SUPABASE_API_URL}")
    logging.info(f"SUPABASE_API_KEY = {SUPABASE_API_KEY[:8]}...")
    if HF_TOKEN:
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN
except Exception as e:
    logging.error(f"Erro CRÍTICO ao carregar variáveis de ambiente: {e}", exc_info=True)
    raise

class AIBot:

    SYSTEM_TEMPLATE = """
Você é a 'FinAssist', uma assistente virtual amigável para controle financeiro pessoal.

Responda no formato:
- "Ok! Registrei suas movimentações. Você gastou R$ {total_spent_final:.2f} ao todo e ainda te sobrou R$ {saldo_final:.2f} para o resto do mês. 😊"
- "Consultando aqui... seu saldo atual é R$ {saldo_final:.2f}. 👍"

Valores:
- Total Gasto: {total_spent_final:.2f}
- Saldo Final: {saldo_final:.2f}
- Gasto na Mensagem: {spent_in_message:.2f}
- Recebido na Mensagem: {received_in_message:.2f}
"""

    def __init__(self):
        try:
            logging.info("Inicializando AIBot com LLM e Supabase...")
            pipe = pipeline(
                "text-generation",
                model="microsoft/phi-2",
                device_map="auto",
                max_new_tokens=150,
                temperature=0.6,
                do_sample=True,
                pad_token_id=50256
            )
            self.__llm = HuggingFacePipeline(pipeline=pipe)
            logging.info("Pipeline LLM local pronto e envolvido por LangChain.")
            self.supabase: Client = create_client(SUPABASE_API_URL, SUPABASE_API_KEY)
            logging.info("Cliente Supabase inicializado.")
        except Exception as e:
            logging.error(f"Erro CRÍTICO ao inicializar AIBot: {e}", exc_info=True)
            raise

    def _get_or_create_client_financial_state(self, client_id: str) -> dict:
        try:
            response = self.supabase.table('Clients') \
                .select('*') \
                .eq('client_id', client_id) \
                .limit(1) \
                .execute()
            logging.info(f"Supabase _get_or_create_client_financial_state (select) response for {client_id}: {response}")

            if response.data:
                data = response.data[0]
            else:
                logging.info(f"Cliente {client_id} não encontrado. Criando nova entrada...")
                initial_data = {
                    'client_id': client_id,
                    'total_spent': 0.0,
                    'total_received': 0.0,
                    'message_text': 'Usuário criado.'
                }
                insert_response = self.supabase.table('Clients').insert(initial_data).execute()
                logging.info(f"Supabase _get_or_create_client_financial_state (insert) response for {client_id}: {insert_response}")
                if insert_response.data:
                    data = insert_response.data[0]
                else:
                    data = initial_data
            return data

        except Exception as e:
            logging.error(f"Supabase Error em _get_or_create_client_financial_state para client_id {client_id}: {e}", exc_info=True)
            return {'client_id': client_id, 'total_spent': 0.0, 'total_received': 0.0, 'message_text': ''}

    def _update_client_data(self, client_id: str, user_message: str, bot_message: str):
        try:
            response = self.supabase.table('Clients') \
                .select('message_text') \
                .eq('client_id', client_id) \
                .limit(1) \
                .execute()

            if response.data:
                current_history = response.data[0].get('message_text', '')
            else:
                current_history = ''

            new_history = f"{current_history}\nUser: {user_message}\nBot: {bot_message}".strip()

            update_data = {'message_text': new_history}
            response = self.supabase.table('Clients') \
                .update(update_data) \
                .eq('client_id', client_id) \
                .execute()

            if hasattr(response, 'error') and response.error:
                logging.error(f"Erro ao atualizar histórico para client_id {client_id}: {response.error}")
            else:
                logging.info(f"Histórico atualizado com sucesso para client_id {client_id}.")

        except Exception as e:
            logging.error(f"Erro ao atualizar histórico para client_id {client_id}: {e}", exc_info=True)

    def _extract_financial_transactions(self, text: str) -> tuple[float, float]:
        text = text.lower()
        money_pattern = r'\b(\d{1,3}(?:\.?\d{3})*(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)\b'
        expense_keywords = ['gastei', 'paguei', 'pagar', 'boleto', 'boletos', 'conta', 'contas', 'despesa', 'custou', 'comprei']
        income_keywords = ['recebi', 'salário', 'salario', 'ganhei', 'depósito', 'reembolso', 'vendi', 'receita']

        total_spent_in_message = 0.0
        total_received_in_message = 0.0

        for match in re.finditer(money_pattern, text):
            value_str = match.group(1)
            try:
                value = float(value_str.replace('.', '').replace(',', '.'))
                context = text[max(0, match.start() - 20):match.end() + 20]
                if any(kw in context for kw in expense_keywords):
                    total_spent_in_message += value
                elif any(kw in context for kw in income_keywords):
                    total_received_in_message += value
            except ValueError:
                continue

        return total_spent_in_message, total_received_in_message

    def invoke(self, client_id: str, question_text: str) -> str:
        if not client_id:
            logging.error("Invoke chamado sem client_id.")
            return "Opa! Preciso saber quem você é para registrar direitinho. 🤔"
        if not self.__llm:
            logging.error("LLM não foi inicializado.")
            return "Desculpe, estou com um problema interno (LLM não carregado)."

        logging.info(f"Invoke iniciado para client_id='{client_id}', question='{question_text}'")

        client_financial_state = self._get_or_create_client_financial_state(client_id)
        current_total_spent = client_financial_state.get('total_spent', 0.0)
        current_total_received = client_financial_state.get('total_received', 0.0)

        spent_in_message, received_in_message = self._extract_financial_transactions(question_text)
        new_total_spent_accumulated = current_total_spent + spent_in_message
        new_total_received_accumulated = current_total_received + received_in_message
        new_balance = new_total_received_accumulated - new_total_spent_accumulated

        bot_response = f"Desculpe, não consegui processar uma resposta para: '{question_text}'"
        try:
            prompt_text = self.SYSTEM_TEMPLATE.format(
                total_spent_final=new_total_spent_accumulated,
                saldo_final=new_balance,
                spent_in_message=spent_in_message,
                received_in_message=received_in_message
            )
            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', prompt_text),
                    MessagesPlaceholder(variable_name='messages'),
                ]
            )
            chain = chat_prompt | self.__llm
            current_conversation_messages = [HumanMessage(content=question_text)]

            response_content = chain.invoke({'messages': current_conversation_messages})
            bot_response = response_content.strip()

        except Exception as e:
            logging.error(f"Erro ao invocar o LLM para {client_id}: {e}", exc_info=True)
            bot_response = f"Ok! Registrei suas movimentações. Você gastou R$ {new_total_spent_accumulated:.2f} ao todo e ainda te sobrou R$ {new_balance:.2f} para o resto do mês. 😊"

        self._update_client_data(client_id, question_text, bot_response)
        return bot_response