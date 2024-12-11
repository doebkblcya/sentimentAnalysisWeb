from django.shortcuts import render
from django.http import JsonResponse
from .nlp.dict.dict import sentiment_analyzer

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        result = sentiment_analyzer.analyze(text)
        return JsonResponse(result)
        
    return render(request, 'analyzer/index.html')