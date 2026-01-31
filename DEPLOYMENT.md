# Streamlit Cloud Deployment Guide

## 🚀 Quick Deployment Steps

### 1. **Repository Setup**
Make sure your GitHub repository contains:
- `streamlit_app.py` (main app file)
- `spam.csv` (dataset)
- `requirements.txt` (dependencies)
- `packages.txt` (system packages)

### 2. **Deploy to Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository
4. Set main file path: `streamlit_app.py`
5. Click "Deploy"

### 3. **Common Issues & Solutions**

#### ❌ **NLTK Data Error**
```
LookupError: Resource punkt not found
```
**Solution:** The app automatically downloads NLTK data using `@st.cache_data`

#### ❌ **File Not Found Error**
```
FileNotFoundError: spam.csv not found
```
**Solution:** Ensure `spam.csv` is in your repository root directory

#### ❌ **Memory Issues**
```
Memory limit exceeded
```
**Solution:** The app uses caching to minimize memory usage

#### ❌ **Import Errors**
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution:** Check `requirements.txt` has all dependencies

### 4. **File Structure for Deployment**
```
your-repo/
├── streamlit_app.py      # Main app (required)
├── spam.csv             # Dataset (required)
├── requirements.txt     # Python dependencies
├── packages.txt         # System packages
└── README.md           # Documentation
```

### 5. **Environment Variables (if needed)**
If you need environment variables:
1. Go to your app settings in Streamlit Cloud
2. Add secrets in the "Secrets" section
3. Access in code: `st.secrets["your_key"]`

### 6. **Performance Optimization**
- Uses `@st.cache_resource` for model training
- Uses `@st.cache_data` for NLTK downloads
- Minimal memory footprint
- Fast loading with caching

### 7. **Testing Locally**
Before deployment, test locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 8. **Troubleshooting Deployment**
If deployment fails:
1. Check the logs in Streamlit Cloud
2. Verify all files are in repository
3. Check requirements.txt syntax
4. Ensure spam.csv is not in .gitignore

## ✅ **Deployment Checklist**
- [ ] Repository has streamlit_app.py
- [ ] spam.csv is in repository root
- [ ] requirements.txt is complete
- [ ] packages.txt exists
- [ ] App runs locally
- [ ] No sensitive data in code
- [ ] Repository is public (for free tier)

## 🔗 **Useful Links**
- [Streamlit Cloud](https://share.streamlit.io)
- [Streamlit Docs](https://docs.streamlit.io)
- [Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)

Your app should now deploy successfully! 🎉