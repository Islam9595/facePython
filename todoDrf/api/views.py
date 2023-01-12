import json
from rest_framework.decorators import api_view
import face_recognition
from opencv.fr import FR
from django.http import JsonResponse


@api_view(['GET'])
def compareFaces(request):
    imgElon = face_recognition.load_image_file('ImagesAttendance/amr.jpg')
    imgTest = face_recognition.load_image_file('ImagesAttendance/amr2.jpg')
    encodeElon = face_recognition.face_encodings(imgElon)[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]
    tolerance = 0.6
    similarly = face_recognition.face_distance([encodeElon], encodeTest)
    results = similarly <= tolerance
    return JsonResponse({
        'data':
            {
                'result': json.dumps(bool(results)),
                'similarly': json.dumps((1-float(similarly))*100)
            }

    })
