import streamlit as st
from google import genai
from google.genai.errors import APIError
from PIL import Image
import io

# --- 1. الإعدادات الأساسية ---

# تعيين عنوان الصفحة والرمز التعبيري
st.set_page_config(page_title="مُعلِّم الرياضيات والفيزياء الذكي", layout="wide")

# تهيئة الاتصال بـ Gemini API
# يستدعي المفتاح تلقائيًا من ملف .streamlit/secrets.toml
try:
    client = genai.Client()
except Exception as e:
    st.error(f"خطأ في تهيئة Google Gemini API. تأكد من إعداد ملف secrets.toml. الخطأ: {e}")
    client = None

# --- 2. دالة بناء التوجيه (Prompt) ---

def build_custom_prompt(question_text, language, response_length, level):
    """
    يبني التوجيه المُركَّب (Prompt) لتوجيه سلوك النموذج.
    """
    # دور المعلم المغربي (الشخصية الأساسية)
    persona = (
        "أنت أستاذ رياضيات وفيزياء مغربي متميز. طريقة شرحك تعتمد على المنهجية المغربية "
        "المتبعة في الثانويات المغربية (باك، علوم رياضية). "
        "يجب أن تكون إجابتك تعليمية، خطوة بخطوة، وتستخدم مصطلحات المنهج."
    )

    # طلبات التخصيص من المستخدم
    customization_rules = (
        f"المستوى الدراسي للطالب: **{level}**.\n"
        f"اللغة المطلوبة للإجابة: **{language}**.\n"
        f"طول الشرح المطلوب: **{response_length}**.\n"
    )

    # توجيه المهمة
    task_instruction = (
        "حل المسألة الرياضية أو الفيزيائية المرفقة (نص أو صورة). "
        "ابدأ بعبارة تشجيعية، ثم قدّم الحل المُفصَّل وفقاً للقيود المذكورة. "
        "المسألة هي: "
    )
    
    # دمج كل شيء
    full_prompt = f"{persona}\n\n---\n\n{customization_rules}\n\n---\n\n{task_instruction}\n{question_text}"
    return full_prompt

# --- 3. دالة معالجة الاستدعاء لـ Gemini ---

def get_gemini_response(prompt, image=None):
    """
    يرسل التوجيه والصورة إلى نموذج Gemini ويستقبل الإجابة.
    """
    if not client:
        return "تعذر الاتصال بخدمة Gemini."

    # تحديد محتوى الإدخال (النص والصورة)
    contents = [prompt]
    if image:
        contents.insert(0, image) # وضع الصورة كأول عنصر

    try:
        # استخدام نموذج multi-modal (قادر على التعامل مع النصوص والصور)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        return response.text
    except APIError as e:
        return f"حدث خطأ في واجهة API: {e}"
    except Exception as e:
        return f"حدث خطأ غير متوقع: {e}"


# --- 4. واجهة Streamlit (UI) ---

st.title("👨‍🏫 مُعلِّم الرياضيات والفيزياء المغربي الذكي")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("تخصيص الإجابة")

    # خيارات التخصيص
    selected_language = st.selectbox(
        "اختر لغة الإجابة:",
        ("العربية الفصحى", "الفرنسية"),
        index=0
    )

    selected_level = st.selectbox(
        "المستوى الدراسي:",
        ("علوم رياضية", "علوم تجريبية", "آداب وعلوم إنسانية", "جدع مشترك علمي"),
        index=0
    )
    
    selected_length = st.select_slider(
        "طول الشرح المطلوب:",
        options=['مختصر', 'متوسط', 'مُفصَّل جداً'],
        value='متوسط'
    )
    
    st.markdown("---")

with col2:
    st.header("إدخال المسألة")
    
    # 1. تحميل صورة
    uploaded_file = st.file_uploader(
        "حمِّل صورة المسألة (مثل تمرين من كتاب أو ورقة):", 
        type=["jpg", "jpeg", "png"]
    )

    # 2. إدخال نص
    text_question = st.text_area(
        "أو اكتب المسألة مباشرة هنا:", 
        height=150, 
        placeholder="أدخل نص المسألة الرياضية أو الفيزيائية..."
    )

    # زر الحل
    solve_button = st.button("✨ اطلب الحل الآن!")

# --- 5. منطق المعالجة ---

if solve_button:
    
    # التحقق من الإدخال
    if not uploaded_file and not text_question.strip():
        st.warning("الرجاء إما تحميل صورة أو كتابة نص المسألة أولاً.")
        st.stop()
        
    # تهيئة المتغيرات
    image_to_send = None
    question_text_input = text_question if text_question.strip() else "تم إرسال المسألة في الصورة المرفقة."

    # معالجة الصورة إذا تم تحميلها
    if uploaded_file is not None:
        try:
            # استخدام مكتبة PIL (Pillow) لتحويل الملف المحمّل إلى كائن صورة
            image_to_send = Image.open(uploaded_file)
            
            # يمكنك عرض الصورة للمستخدم كتأكيد
            st.sidebar.image(image_to_send, caption="الصورة التي تم تحميلها", use_column_width=True)
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصورة: {e}")
            image_to_send = None

    # بناء التوجيه النهائي
    final_prompt = build_custom_prompt(
        question_text_input, 
        selected_language, 
        selected_length, 
        selected_level
    )
    
    # عرض حالة المعالجة
    with st.spinner("🧠 الذكاء الاصطناعي يُعالج المسألة ويُعِد الشرح..."):
        # الحصول على الرد
        response_text = get_gemini_response(final_prompt, image_to_send)
        
    # عرض النتيجة
    st.header("✅ الحل والشرح المُفصَّل")
    st.success(f"**المستوى:** {selected_level} | **اللغة:** {selected_language} | **الطول:** {selected_length}")
    st.markdown(response_text)


# --- 6. كيفية التشغيل ---

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**طريقة التشغيل:**\n"
    "1. تأكد من حفظ مفتاح API في ملف `.streamlit/secrets.toml`.\n"
    "2. قم بتشغيل التطبيق من Terminal بعد تفعيل البيئة الافتراضية:\n"
    "```bash\n"
    "streamlit run app.py\n"
    "```"
)