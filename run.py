from flask import Flask, request, jsonify, send_from_directory
import os
import MySQLdb
import shutil
import json
import cv2
import subprocess
# import torch # ---- 주석 해제

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # 유니티 객체 감지

app.config['RESULT_FOLDER'] = 'results' # 업로드 결과 저장
app.config['BACKGROUND_FOLDER'] = 'background'

app.config['XRAY_FOLDER'] = 'images_xray' # 엑스레이 이미지

app.config['DEFECT_FOLDER'] = 'defects' # 결함 검사 결과

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'image_db'


def get_db_connection():
    return MySQLdb.connect(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        passwd=app.config['MYSQL_PASSWORD'],
        db=app.config['MYSQL_DB']
    )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def update_detect_status(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE images SET detect = TRUE WHERE filename = %s",
            (filename,)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        return str(e)
    finally:
        cursor.close()
        conn.close()
    
    return 'Update successful'

def move_to_defects_from_results():
    source_folder = './yolov3/result'
    destination_folder = app.config['DEFECT_FOLDER']

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(destination_folder, filename))

    return 'Move successful'

def is_image_file(filename):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    return os.path.splitext(filename)[1].lower() in image_extensions

def update_defect_status():
    files2 = [os.path.join(app.config['RESULT_FOLDER'], f) for f in os.listdir(app.config['RESULT_FOLDER']) if os.path.isfile(os.path.join(app.config['RESULT_FOLDER'], f))]
    files = [os.path.join(app.config['DEFECT_FOLDER'], f) for f in os.listdir(app.config['DEFECT_FOLDER'])
             if os.path.isfile(os.path.join(app.config['DEFECT_FOLDER'], f)) and is_image_file(f)]
    
    if files:
        latest_file2 = max(files2, key=os.path.getmtime)
        filename2 = os.path.basename(latest_file2)

        latest_file = max(files, key=os.path.getmtime)
        defect_filename = os.path.basename(latest_file)
        defect_filepath = latest_file

        update_sql = """
        UPDATE images
        SET defect_filename = %s, defect_filepath = %s
        WHERE filename = %s;
        """

        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(update_sql, (defect_filename, defect_filepath, filename2))
            conn.commit()
        except Exception as e:
            conn.rollback()
            return str(e)
        finally:
            cursor.close()
            conn.close()
        
        return 'Update successful'

# @app.route('/cuda', methods=['GET'])
# def cuda():
#     print(torch.__version__)  # PyTorch 버전 확인
#     print(torch.version.cuda)  # CUDA 버전 확인
#     print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인
    
#     return 'cuda check', 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Save file info to MySQL
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images (filename, filepath) VALUES (%s, %s)", (filename, filepath))
        conn.commit()
        cursor.close()
        conn.close()

        return 'File uploaded successfully', 200
    else:
        return 'Invalid file type', 400


# 유니티에서 1초마다 찍은 카메라 캡쳐본을 저장한 DB에 연결해 최신 이미지 가져옴
@app.route('/images/latest', methods=['GET'])
def get_image():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, filepath FROM images ORDER BY id DESC LIMIT 1")
    image = cursor.fetchone()
    cursor.close()
    conn.close()

    if image:
        filename, filepath = image
        if not os.path.exists(app.config['RESULT_FOLDER']):
            os.makedirs(app.config['RESULT_FOLDER'])
        
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)

        try:
            shutil.copy(filepath, result_path)
            return f"Image saved to {result_path}", 200
        except Exception as e:
            return str(e), 500
    else:
        return 'Image not found', 404


@app.route('/images/contour', methods=['GET'])
def detect_contour():
    files = [os.path.join(app.config['RESULT_FOLDER'], f) for f in os.listdir(app.config['RESULT_FOLDER']) if os.path.isfile(os.path.join(app.config['RESULT_FOLDER'], f))]
    if not files:
        return 'Latest image not found', 404

    latest_file = max(files, key=os.path.getmtime)

    if latest_file and os.path.exists(latest_file):
        background_image_path = os.path.join(app.config['BACKGROUND_FOLDER'], 'conveyor_img.png')

        background_image = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)
        background_image = cv2.resize(background_image, (500, 500))
        object_image = cv2.imread(latest_file, cv2.IMREAD_GRAYSCALE)
        object_image = cv2.resize(object_image, (500, 500))

        diff = cv2.absdiff(background_image, object_image)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 컨투어가 감지되었는지 확인
        if contours:
            # 가장 큰 컨투어 찾기
            cnt = max(contours, key=cv2.contourArea)

            # 최소 면적 기준으로 필터링
            if cv2.contourArea(cnt) > 500:
                # 객체의 바운딩 박스 계산
                x, y, w, h = cv2.boundingRect(cnt)
                object_center = (x + w // 2, y + h // 2)  # 객체 중심 좌표
                #print(object_center)
                
                # 객체 중심이 이미지 중심에 가까운지 확인 (허용 오차 50 픽셀)
                tolerance = 5
                image_center = (background_image.shape[1] // 2, object_image.shape[0] // 2)
                #print(image_center)
                if abs(object_center[1] - image_center[1]) < tolerance:
                    filename = os.path.basename(latest_file)
                    update_status_result = update_detect_status(filename)
                
                    if update_status_result == 'Update successful':
                        return 'Object detected DB update', 200
                    else:
                        return update_status_result, 500
                else:
                    return 'Object not in center', 404
        else:
            return 'Object detect no contour', 404
    else:
        return 'Latest image not found', 404

# @app.route('/detect/defect', methods=['GET'])
# def detect_contour():

@app.route('/run_detection', methods=['POST'])
def run_detection():
    xray_folder = app.config['XRAY_FOLDER']
    processed_record_path = 'processed_files.json'

    # Initialize processed files list
    processed_files = []

    # Try to load the processed files record
    if os.path.exists(processed_record_path):
        try:
            with open(processed_record_path, 'r') as f:
                processed_files = json.load(f)
        except json.JSONDecodeError:
            # Handle the case where the JSON is invalid or empty
            processed_files = []
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error loading processed files record: {str(e)}"}), 500

    # Get the list of files in the XRAY_FOLDER
    files = [os.path.join(xray_folder, f) for f in os.listdir(xray_folder) if os.path.isfile(os.path.join(xray_folder, f))]
    
    # Filter out the processed files
    unprocessed_files = [f for f in files if f not in processed_files]
    
    if not unprocessed_files:
        return jsonify({"status": "error", "message": "No unprocessed X-ray images found"}), 404

    oldest_file = min(unprocessed_files, key=os.path.getmtime)
    processed_files.append(oldest_file)
    with open(processed_record_path, 'w') as f:
        json.dump(processed_files, f)
    # return jsonify({"status": "success", "message": oldest_file}), 200

    # if(move_to_defects_from_results() == 'Move successful' and update_defect_status() == 'Update successful'):
    #     return jsonify({"status": "success", "message": oldest_file}), 200
    # return jsonify({"status": "error", "message": 'move failed'}), 400

    try:
        # 명령어를 실행
        command = [
            'python', './yolov3/detect_test.py', 
            '--weights', './yolov3/weights/last.pt', 
            '--source', oldest_file, #'./yolov3/images/002_20200922_083747(7).jpg', #'./yolov3/images', 
            '--cfg', './yolov3/yolov3-spp.cfg', 
            '--names', './yolov3/classes.names', '--output', './yolov3/result', 
            '--half',
            '--save-txt' 
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # 명령어 실행 결과 확인
        if result.returncode == 0:
            
            defects_check = "defects, Done" in result.stdout

            if(move_to_defects_from_results() == 'Move successful' and update_defect_status == 'Update successful'):
                return jsonify({"status": "success", "defects_check": defects_check, "output": result.stdout}), 200
        else:
            return jsonify({"status": "error", "error": result.stderr}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)