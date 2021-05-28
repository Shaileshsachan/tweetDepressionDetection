import streamlit as st

from depression import metrics, testData, preds_tf_idf, process_message, sc_tf_idf, bow


def start():
    st.title("Tweet Classifier ML app")
    html_temp = """
    <div style="background-color:black;padding:15px">
    <h2 style="color:white";text-align:center;">Naive Bayes's  Theorem</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    load_css('style.css')
    st.header("Created by...")
    st.text("Shailesh Sachan")
    st.text("Mujtaba Sayyed")
    st.text("Taha Khan")

    message = st.text_input("Enter Tweet")
    if st.button("Predict"):
        result = predict_tweet(message)
        if result == False:
            prediction = 'Non-Depressive Tweet'
            st.success('Tweet was classified as {}'.format(prediction))
            acc = accuracy()
            st.success('Accuracy : {}'.format(acc))
        elif result == True:
            prediction = 'Depressive Tweet'
            st.write("AASRA")
            st.write("http://www.aasra.info")
            st.write("Phone: 91-22-27546669")
            st.write("Phone: 91-22-27546667")
            st.write("Email: aasrahelpline@yahoo.com")
            c_img = 'dep.jpg'
            st.success('Tweet was classified as {}'.format(prediction))
            accc = accuracy()
            st.success('Metrics : {}'.format(accc))
        else:
            st.write("Failed to detect the tweet")


def accuracy():
    return metrics(testData['label'], preds_tf_idf)


def predict_tweet(message):
    pm = process_message(message)
    print(pm)
    return bow.classify(pm)
    # return sc_tf_idf.classify(pm)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


start()
