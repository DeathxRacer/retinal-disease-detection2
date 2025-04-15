# Add this CSS to the existing style block
st.markdown("""
    <style>
    /* Existing styles remain here */

    [data-testid="fileUploader"] {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: #2e2e2e;
        border-radius: 10px;
        padding: 15px;
        max-width: 400px;
        width: 90%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: white;
    }

    [data-testid="fileUploader"] label {
        color: white !important;
        text-align: center;
        font-size: 16px;
    }

    [data-testid="fileUploader"] span {
        color: #00bcd4 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Modified uploader section (remove any existing positioning classes)
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"], 
                                help="Drag and drop your retinal scan here")
