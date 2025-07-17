from flask import Flask, render_template, request, redirect, url_for, g, session, flash, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chat_models import ChatOpenAI

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os, logging
import urllib.parse
import bcrypt
from flask_wtf import FlaskForm
import pymysql
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email
from langchain.prompts import PromptTemplate
import json

# ─── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="mindly_debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Flask & env ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config.update(
    MYSQL_HOST="127.0.0.1",
    MYSQL_PORT=3307,
    MYSQL_USER="root",
    MYSQL_PASSWORD="",
    MYSQL_DB="Mindly",
)

# DB connection
def get_db():
    if 'db' not in g:
        g.db = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
openrouter_api_key = os.getenv('OPENROUTESERVICE_API_KEY')

# ─── Pinecone retriever ──────────────────────────────────────────────────────────
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="mindly",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ─── LLM & prompt ───────────────────────────────────────────────────────────────
# llm = OllamaLLM(model="mistral:7b", temperature=0.4, top_p=0.9, max_tokens=300, repeat_penalty=1.1)
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-0528-qwen3-8b:free",  
    openai_api_key = openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)
system_prompt = (
    "You are Mindly, a kind and empathetic mental health assistant. "
    "Your responses should be helpful, calming, and clearly formatted. "
    "Always follow this format:\n"
    "1. Short point 1\n"
    "2. Short point 2\n"
    "...\n"
    "Limit to **5 numbered points**, each ≤1 sentence. "
    "If you can’t help, say: “I'm here for you, but I may need more information to help with that.”\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ─── Embedder ────────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Auth Forms ─────────────────────────────────────────────────────────────────
class RegisterForm(FlaskForm):
    name     = StringField("Name", validators=[DataRequired()])
    email    = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit   = SubmitField("Register")

class LoginForm(FlaskForm):
    email    = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit   = SubmitField("Login")

# ─── Routes ──────────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data
        
        db = get_db()
        cursor = db.cursor()
        
        try:
            # Check if user already exists
            cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                flash("An account with this email already exists. Please login.", "error")
                return redirect(url_for('login'))

            # Create new user if doesn't exist
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                (name, email, hashed)
            )
            db.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            db.rollback()
            logging.error(f"Registration error: {str(e)}")
            flash("An error occurred during registration. Please try again.", "error")
            return redirect(url_for('register'))
            
        finally:
            cursor.close()

    return render_template("register.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        db = get_db()
        cursor = db.cursor()  # No arguments needed since we set cursorclass in connection
        
        try:
            # Explicit column selection for security and clarity
            cursor.execute(
                "SELECT id, name, email, password FROM users WHERE email = %s", 
                (email,)
            )
            user = cursor.fetchone()
            
            if user and bcrypt.checkpw(password.encode("utf-8"), user['password'].encode("utf-8")):
                session["user_id"] = user['id']
                session["user_name"] = user['name']
                return redirect(url_for("index"))
            
            flash("Invalid email or password")
            return redirect(url_for("login"))
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            flash("An error occurred during login")
            return redirect(url_for("login"))
        finally:
            cursor.close()
    
    return render_template("login.html", form=form)
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/save_message", methods=["POST"])
def save_message():
    if "user_id" not in session:
        return jsonify({"status": "unauthorized"}), 401
    data = request.get_json()
    user_message = data.get("user_message")
    bot_message = data.get("bot_message")
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO messages (user_id, user_message, bot_message) VALUES (%s, %s, %s)",
        (session["user_id"], user_message, bot_message)
    )
    db.commit()
    cursor.close()
    return jsonify({"status": "success"})

# @app.route("/")
# def index():
#     return render_template("chat2.html")
# Update the index route
def generate_mental_health_insights(user_id):
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Get all user messages
        cursor.execute("""
            SELECT user_message FROM messages 
            WHERE user_id = %s 
            ORDER BY created_at ASC
        """, (user_id,))
        messages = cursor.fetchall()
        
        if not messages:
            return None
            
        result = '. '.join(msg['user_message'] for msg in messages) + '.'
        
        # Generate analysis
        prompt_template = PromptTemplate.from_template("""
        You are a mental health assistant. Analyze the following conversation history from a user:
        {user_messages}

        Return a JSON with:
        - anxiety_score (0–100)
        - stress_score (0–100)
        - mental_health_score (0–100)
        - top_keywords (5–10 keywords)
        - recommendations (2 tips to improve well-being)

        Respond only in JSON format.
        """)

        formatted_prompt = prompt_template.format(user_messages=result)
        raw_response = llm.invoke(formatted_prompt)

        try:
            scores = json.loads(raw_response)
            return {
                "user_id": user_id,
                "anxiety_score": scores["anxiety_score"],
                "stress_score": scores["stress_score"],
                "mental_health_score": scores["mental_health_score"],
                "top_keywords": json.dumps(scores["top_keywords"]),
                "recommendations": json.dumps(scores["recommendations"])
            }
        except Exception as e:
            logging.error(f"Error parsing LLM response: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Insight generation error: {str(e)}")
        return None
    finally:
        cursor.close()
@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        db = get_db()
        cursor = db.cursor()

        # Get latest analysis
        cursor.execute("""
            SELECT * FROM mental_health_scores 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (session['user_id'],))
        analysis = cursor.fetchone()

        if analysis:
            scores = {
                "anxiety_score": analysis["anxiety_score"],
                "stress_score": analysis["stress_score"],
                "mental_health_score": analysis["mental_health_score"],
                "top_keywords": json.loads(analysis["top_keywords"]),
                "recommendations": json.loads(analysis["recommendations"]),
                "last_updated": analysis["created_at"].strftime("%b %d, %Y %H:%M")
            }
            return render_template("dashboard.html", scores=scores)
        else:
            # Check message count for progress
            cursor.execute("""
                SELECT COUNT(*) as message_count 
                FROM messages 
                WHERE user_id = %s
            """, (session['user_id'],))
            count = cursor.fetchone()["message_count"]
            remaining = 5 - (count % 5)
            return render_template("dashboard.html", scores=None, remaining=remaining)
            
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        return render_template("dashboard.html", scores=None)
    finally:
        cursor.close()
@app.route("/")
def index():
    return render_template("landing.html")

# Add new chat route
# @app.route("/chat")
# def chat_route():
#     if "user_id" not in session:
#         return redirect(url_for("login"))
    
#     try:
#         db = get_db()
#         cursor = db.cursor()
#         cursor.execute("""
#             SELECT user_message, bot_message, created_at 
#             FROM messages 
#             WHERE user_id = %s 
#             ORDER BY created_at ASC
#         """, (session["user_id"],))
#         messages = cursor.fetchall()
#         return render_template("chat2.html", messages=messages)
#     except Exception as e:
#         logging.error(f"Error fetching messages: {str(e)}")
#         return render_template("chat2.html", messages=[])
#     finally:
#         cursor.close()

@app.route("/chat")
def chat_route():
    # if "user_id" not in session:
    #     return redirect(url_for("login"))
    
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            SELECT user_message, bot_message, created_at 
            FROM messages 
            WHERE user_id = %s 
            ORDER BY created_at ASC
        """, (session["user_id"],))
        messages = cursor.fetchall()
        return render_template("chat2.html", messages=messages)
    except Exception as e:
        logging.error(f"Error fetching messages: {str(e)}")
        # flash("Error loading chat history")
        return render_template("chat2.html", messages=[])
    finally:
        cursor.close()
# Add this route to your Flask app (app2.py)
@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "user_id" not in session:
        return jsonify({"status": "unauthorized"}), 401
    
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "DELETE FROM messages WHERE user_id = %s",
            (session["user_id"],)
        )
        db.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Error clearing history: {str(e)}")
        return jsonify({"status": "error"}), 500
    finally:
        cursor.close()
@app.route("/speech")
def speech_route():

    
    return render_template("speech_chat.html")
@app.route("/get", methods=["POST"])
def chat():
    if request.method != "POST":
            return jsonify({"error": "Method not allowed"}), 405
    try:
        msg = request.form["msg"]
        decoded_msg = urllib.parse.unquote_plus(msg) # Add this line
        logging.info(f"User input: {msg}")
        docs = retriever.invoke(msg)
        sims = []
        query_emb = embedder.encode(msg, convert_to_tensor=True)
        for d in docs:
            sims.append(util.cos_sim(query_emb, embedder.encode(d.page_content, convert_to_tensor=True)).item())
        avg_sim = sum(sims)/len(sims)
        if avg_sim < 0.45:
            return "I'm here for you, but I may need more information to help with that."
        chain = create_stuff_documents_chain(llm, prompt)
        out = chain.invoke({"input": msg, "context": docs})
        answer = out if isinstance(out, str) else out.get("answer")
        if "user_id" in session:
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO messages (user_id, user_message, bot_message) VALUES (%s, %s, %s)",
                (session["user_id"], decoded_msg, answer)
            )
            db.commit()
            cursor.execute("""
            SELECT COUNT(*) as message_count 
            FROM messages 
            WHERE user_id = %s
        """, (session["user_id"],))
        count = cursor.fetchone()["message_count"]
        
        if count % 5 == 0:
            insights = generate_mental_health_insights(session["user_id"])
            if insights:
                cursor.execute("""
                    INSERT INTO mental_health_scores 
                    (user_id, anxiety_score, stress_score, mental_health_score, top_keywords, recommendations)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    insights["user_id"],
                    insights["anxiety_score"],
                    insights["stress_score"],
                    insights["mental_health_score"],
                    insights["top_keywords"],
                    insights["recommendations"]
                ))
            
            db.commit()
            cursor.close()
        return answer
    except Exception as e:
        logging.error(f"Chat error: {e}", exc_info=True)
        return "Sorry, something went wrong. Please try again."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
