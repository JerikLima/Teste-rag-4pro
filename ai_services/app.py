import streamlit as st
import time
from ai_bot import AIBot
import logging
import uuid 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource 
def load_bot():
    logging.info("Tentando carregar o AIBot...")
    try:
        bot_instance = AIBot()
        logging.info("AIBot carregado com sucesso.")
        return bot_instance
    except Exception as e:
        logging.error(f"Falha ao carregar AIBot: {e}", exc_info=True)
        st.error(f"Erro crítico ao carregar o assistente: {e}")
        return None

bot = load_bot()

if bot is None:
    st.warning("O assistente não pôde ser inicializado. A aplicação não pode continuar.")
    st.stop()

if 'client_id' not in st.session_state:

    st.session_state.client_id = f"st_user_{str(uuid.uuid4())[:8]}"
    logging.info(f"Novo client_id de sessão gerado: {st.session_state.client_id}")

client_id = st.session_state.client_id

st.title(f"💬 FinAssist - Sua Assistente Financeira")

st.caption("Digite suas despesas, receitas ou faça perguntas financeiras.")

if "messages" not in st.session_state: 
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Como posso te ajudar com suas finanças hoje?"})


# Exibe o histórico do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
user_input = st.chat_input("Digite sua mensagem aqui...")

if user_input:
    logging.info(f"Usuário ({client_id}) input: {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Processando sua mensagem..."):
        start_time = time.time()
        try:
            bot_reply = bot.invoke(client_id=client_id, question_text=user_input)
            processing_time = time.time() - start_time
            logging.info(f"Resposta do bot ({client_id}) gerada em {processing_time:.2f} segundos: {bot_reply}")

            if bot_reply:
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                with st.chat_message("assistant"):
                    st.markdown(bot_reply)
            else:
                error_message = "Desculpe, não consegui gerar uma resposta desta vez."
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                     st.warning(error_message)
                logging.warning(f"Bot retornou uma resposta vazia ou None para {client_id}.")

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ocorreu um erro ao processar sua mensagem ({client_id}) após {processing_time:.2f}s: {e}"
            st.error(error_msg)
            logging.error(error_msg, exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": f"Desculpe, ocorreu um erro técnico ao processar: {e}"})
            st.rerun()