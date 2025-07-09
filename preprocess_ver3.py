import os

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
import cv2  # 기존 cv
import numpy as np
from pathlib import Path  # 파일 디렉토리 라이브러리
import pandas as pd
from sklearn.model_selection import train_test_split  # 테스트셋 분리 라이브러리
import insightface
from insightface.app import FaceAnalysis  # 얼굴 추적
import albumentations as A  # 증강된 데이터 tensor 형태로 변환
from albumentations.pytorch import ToTensorV2  # 텐서 변환
import torch
from tqdm import tqdm  # 진행 바 표시
import json  # json 파일 저장 및 형식 활용
import logging  # log 관련 라이브러리
from concurrent.futures import ThreadPoolExecutor, as_completed  # 병렬 라이브러리
import warnings
import pickle  # 직렬화 저장
import webdataset as wds  # webdataset 속도증가
import torchvision
from torchvision import transforms  # 이미지 변환 함수 모음
from sys import exc_info

warnings.filterwarnings("ignore")


class DeepFake_Data_Preprocessing:
    def __init__(self, config):
        self.config = config  # default 설정
        self.setup_logging()  # logging 설정
        self.setup_insightface()  # insightface 라이브러리 설정
        self.setup_augmentation()  # 데이터 증강 설정

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,  # 로그 출력 범위 설정(info부터)
            format="%(asctime)s - %(levelname)s - %(message)s",  # 기록시간, 출력층, 로그메세지 표현
            handlers=[
                # 로그 출력할 핸들러 목록 직접 지정
                logging.FileHandler("processing.log", mode="w"),  # 파일에 로그 기록
                logging.StreamHandler()  # 표준출력 로그, 콘솔 창 로그 확인가능
            ]
        )
        self.logger = logging.getLogger(__name__)  # 인스터스

    # insightface 인스턴스 설정
    def setup_insightface(self):
        try:
            self.face_app = FaceAnalysis(
                name='buffalo_l',  # buffalo_l 모델 로드
                # FaceAnalysis 인스턴스
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],  # Cuda or cpu
                allowed_modules=["detection"]  # 얼굴 검출 모델만 로드
            )
            # 모델 실행 초기화설정
            self.face_app.prepare(
                ctx_id=self.config.get("gpu_id", 0),  # gpu 사용 설정
                # 얼굴 검출 해상도 설정 (640x640)-Default
                # 높은 해상도 사용 시 추후 정확도 증가/ 처리시간 증가
                det_size=self.config.get("det_size", (640, 640)),
                det_thresh=0.4
            )
            self.logger.info("insightface 모델 로드 성공")
        except Exception as e:
            self.logger.error(f"insightface 모델로드 실패{e}")
            raise

    # 데이터 증강 인스턴스
    def setup_augmentation(self):
        # 이미지 증강 파이프라인(훈련데이터)
        self.train_transform = A.Compose([
            # 이미지크기를 config 설정 이미지 크기로 변환
            A.Resize(height=self.config["image_size"], width=self.config["image_size"]),
            A.OneOf([
                # 선택시 가우시안, 모션 블러 설정
                A.GaussianBlur(p=1.0), A.MotionBlur(p=1.0),
            ], p=0.2),  # 나열된 블러 중 하나를 20% 확률로 적용 - 커스텀 가능
            A.OneOf([
                # 밝기,대비, 색조,채도,명도 조절
                A.RandomBrightnessContrast(p=1.0), A.HueSaturationValue(p=1.0),
            ], p=0.2),  # 20% 확률- 커스텀 가능
            A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우반전 - 커스텀 가능
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 픽셀 값을 [-1,1]로 정규화
            ToTensorV2()  # 텐서형태로 변환(C*H*W) 채널*height*width
        ])
        # 검증/테스트 데이터증강 파이프라인
        self.val_transform = A.Compose([
            A.Resize(height=self.config["image_size"], width=self.config["image_size"]),
            # RGB 채널 별로 [-1,1]로 정규화, 검증 단계: 증강없이 분포만
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()  # 텐서형태로 변환(C*H*W) 채널*height*width
        ])

    def ext_face_from_frame(self, frame):
        # 프레임에서 얼굴 검출(crop)
        try:
            faces = self.face_app.get(frame)  # FaceAnalysis로 검출된 얼굴결과 리스트 가져오기
            extracted_faces = []  # 자른 얼굴 이미지, 좌표 저장리스트
            for face in faces:  # 얼굴 객체 순회
                bbox = face.bbox.astype(int)  # 얼굴 좌표 정수화
                margin = self.config.get("face_margin", 20)  # 얼굴영역 마진 설정(설정값)/ Deafault:20
                h, w = frame.shape[:2]  # 원본 프레임 h,w
                x1, y1, x2, y2 = bbox  # 바운딩 박스 좌표 좌상단(x1,y1) 우하단(x2,y2)로 분해
                # 클램핑: 좌표값이 음수일 경우 화면을 벗어나는 경우 발생, 값 지정으로 방지
                x1, y1 = max(0, x1 - margin), max(0, y1 - margin)  # 마진 만큼 좌상단 확대 최소값 0으로 클램핑
                x2, y2 = min(w, x2 + margin), min(h, y2 + margin)  # 우하단 동일
                face_img = frame[y1:y2, x1:x2]  # 보정된 좌표값에 따라 원본 프레임에서 얼굴 crop
                if face_img.size > 0:  # 잘라낸 이미지가 있을때만
                    # 자른얼굴, 좌표값 딕셔너리로 저장
                    extracted_faces.append({"face_img": face_img, "bbox": [x1, y1, x2, y2]})
            return extracted_faces  # 얼굴이미지 목록, 좌표 반환
        except Exception:
            return []  # 오류 시 빈 리스트 반환

    def extract_frames_from_video(self, video_path, max_frames=None):
        # 영상 프레임 추출 코드
        frames = []  # 프레임 저장 할 리스트 초기화
        cap = cv2.VideoCapture(str(video_path))  # cv2 캡쳐
        if not cap.isOpened(): return frames  # 비디오 파일 로드 실패 빈 배열 반환
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오 전체 프레임 정수화
        if total_frames == 0:  # 프레임 0이면 비디오X 캡쳐 종료
            cap.release()
            return frames
        # 최대 프레임(max_frame) 설정, 전체프레임 수가 크면, 전체/최대 한 정수값을 정해 프레임 균등하게 추출
        # else 모든 프레임 처리
        step = total_frames // max_frames if max_frames and total_frames > max_frames else 1
        frame_count = 0  # 프레임 카운터
        try:
            # 프레임 수가 최대 설정 프레임보다 작거나, 설정 프레임 수가 None 이면 무한반복
            while len(frames) < (max_frames or float('inf')):
                # 프레임 수 로드 성공 시 ret, frame 으로 받음
                ret, frame = cap.read()
                if not ret: break
                # 지정된 간격에 해당되는 프레임만 샘플링
                if frame_count % step == 0:
                    # 기본 BGR -> RGB로 저장
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
        finally:
            cap.release()
        return frames  # 추출된 RGB 프레임 리스트 반환

    # iou: 두 박스 겹치는 정도 수치로 나타내는거 1에 가까우면 거의 완전히 겹친거, 0에 가까우면 거의 안 겹친거
    # 추가 이유: 검출 박스와 정답 박스를 비율 측정해서 모델 정확도 높임
    def calculate_iou(self, boxA, boxB):
        # 교집합 좌상단/우하단 설정
        xA = max(boxA[0], boxB[0]);
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]);
        yB = min(boxA[3], boxB[3])
        # 교집합 면적 설정 (음수 방지 클램핑)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        # 각 박스의 전체면적 계산
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        # Iou 공식:교집합 면적)/(박스A 면적 + 박스B 면적 − 교집합 면적)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    # track: 연속된 프레임에서 "같은객체"가 검출된 모든 정보 묶음관리
    # 매칭: 트랙과 현재 프레임에서 검출된 얼굴 박스를 묶음
    # 매칭이 필요한 이유: 연속된 프레임에서 얼굴에 일관된 id를 부여하기 위해
    def track_face_over_frames(self, frame_faces, iou_threshold=0.4):  # - 커스텀 가능 iou_threshold=0.4
        tracked_seq = []  # 얼굴 트랙(시퀀스) 정도 리스트 초기화
        next_face_id = 0  # id 카운터

        # 첫번쨰 프레임에서 초기 트랙생성
        if frame_faces and frame_faces[0]:  # 첫번째 프레임에서 얼굴이 있다면 진행
            for face in frame_faces[0]:
                tracked_seq.append({
                    "id": next_face_id,  # 트랙 고유번호
                    "seq": [face["face_img"]],  # 해당 얼굴이 매칭된 프레임들의 이미지(리스트)
                    "last_bbox": face['bbox'],  # 마지막으로 매칭된 얼굴 bbox
                    'active': True})  # 현재 프래임에서 매칭 성공여부 플래그
                next_face_id += 1  # 각 얼굴에 고유 ID를 부여하기 위해 1씩 증가
        # 두번쨰 프레임부터 마지막끼지 트랙 매칭
        for i in range(1, len(frame_faces)):
            cur_frame_faces = frame_faces[i]
            for track in tracked_seq: track["active"] = False  # 기존 플래그 off, 매칭 전 상태로 표시
            matched_face_in_cur_frame = [False] * len(cur_frame_faces)  # 얼굴 매칭 기록 리스트
            # 순회 매칭 시도
            for track in tracked_seq:
                if not track.get("last_bbox"): continue
                best_iou, best_match_idx = -1, -1  # 가장 높은 iou값과, 인덱스 초기화
                # 현재 프레임 얼굴리스트 순회
                for j, face_info in enumerate(cur_frame_faces):
                    if matched_face_in_cur_frame[j]: continue
                    # 이전 프레임과 현재 프레임 박스 사이에 iou 계산
                    iou = self.calculate_iou(track["last_bbox"], face_info["bbox"])
                    # 임계값 이상이면서 최대 iou보다 up, 매칭 후보 업데이트
                    if iou > iou_threshold and iou > best_iou:
                        best_iou, best_match_idx = iou, j
                # 하나라도 임계값 만족 얼굴 존재 시 트랙갱신
                if best_match_idx != -1:
                    track['seq'].append(cur_frame_faces[best_match_idx]['face_img'])  # 트랙 시퀀스 추가
                    track['last_bbox'] = cur_frame_faces[best_match_idx]['bbox']  # 트랙 마지막 bbox 갱신
                    track['active'] = True  # 매칭 플래그
                    matched_face_in_cur_frame[best_match_idx] = True  # 중복매칭 방지
            # 매칭안된 얼굴 신규트랙 생성
            for j, matched in enumerate(matched_face_in_cur_frame):
                if not matched:
                    face_info = cur_frame_faces[j]  # 새로운 트랙 얼굴정보 갱신
                    # 리스트 업데이트
                    tracked_seq.append(
                        {"id": next_face_id, "seq": [face_info['face_img']], 'last_bbox': face_info['bbox'],
                         'active': True})
                    next_face_id += 1
        # 최종반환 얼굴 시퀀스 길이 설정
        min_seq_length = self.config.get("seq_length", 16) // 2
        final_seqs = [track['seq'] for track in tracked_seq if len(track['seq']) >= min_seq_length]
        return final_seqs

    # faces 리스트 target_length 로 변환
    def adjust_seq_length(self, faces, target_length):
        cur_length = len(faces)
        if cur_length == 0: return []
        if cur_length == target_length:
            return faces
        # 시퀀스 길이 줄어야 할 떄
        elif cur_length > target_length:
            # 균등한 간격으로 인덱스 분할, 균등 샘플링
            indices = np.linspace(0, cur_length - 1, target_length, dtype=int)
            # 계산된 인덱스 순서대로 length를 줄인 새로운 시퀀스 반환
            return [faces[i] for i in indices]
        else:
            # 부족한 length 채워서 target_length로 맞춤
            return faces + [faces[-1]] * (target_length - cur_length)

    # 비디오 정보 저장 함수
    def process_video_and_ext_faces(self, video_info, output_dir):
        # 각 비디오 경로, 레이블, 고유 id
        video_path, label, video_id = video_info
        # 결과 pickle 파일로 저장
        output_path = output_dir / f"{video_id}.pkl"
        # 이미 처리된 파일 존재 시 캐시 재사용
        if output_path.exists():
            try:
                # 피클파일 바이러니 모드 열고 읽기, 저장된 결과 메모리 로드
                with open(output_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        try:
            # 1단계: 프레임 추출 로그
            frames = self.extract_frames_from_video(video_path, max_frames=self.config.get("max_fps", 16))
            if not frames:
                self.logger.warning(f"[{video_id}] 비디오에서 프레임을 추출하지 못했습니다. 건너뜁니다.")
                return []

            # 2단계: 얼굴 검출 로그
            frames_faces = [self.ext_face_from_frame(frame) for frame in frames]

            # 비디오 전체에서 검출된 얼굴이 하나도 없는 경우
            if all(len(f) == 0 for f in frames_faces):
                self.logger.warning(f"[{video_id}] 비디오의 어떤 프레임에서도 얼굴을 검출하지 못했습니다. 건너뜁니다.")
                return []

            # 3단계: 얼굴 트래킹 로그
            tracked_faces = self.track_face_over_frames(frames_faces)
            if not tracked_faces:
                self.logger.warning(f"[{video_id}] 검출된 얼굴들을 트래킹하지 못했습니다. (시퀀스 길이 부족). 건너뜁니다.")
                return []

            # 모든 단계를 통과한 성공 사례
            self.logger.info(f"[{video_id}] 처리 성공: {len(tracked_faces)}개의 얼굴 시퀀스 생성.")

            video_res = []  # 비디오 최종결과 리스트(얼굴 시퀀스, 메타정보 etc)
            for i, face_seq in enumerate(tracked_faces):  # 트래킹된 얼굴 시퀀스 순회
                adjusted_seq = self.adjust_seq_length(face_seq, self.config.get("seq_length", 16))  # 설정 길이에 맞게 자르거나 패딩
                if not adjusted_seq: continue  # 길이 조정 후 시퀀스 비어 있으면 트랙 패스
                video_res.append({
                    "video_id": f"{video_id}_face_{i}",
                    "faces": adjusted_seq, "label": label,
                    "original_video_path": str(video_path)  # 'orgin_video_path' -> 'original_video_path' 수정
                })
            with open(output_path, "wb") as f:
                # 리스트 형태 video_res 직렬 저장
                pickle.dump(video_res, f)
            return video_res
        except Exception as e:
            # 4단계: 예상치 못한 에러 로그
            self.logger.error(f"[{video_id}] 처리 중 예상치 못한 오류 발생: {e}", exc_info=False)
            return []

    def create_dataset_metadata(self, data_dir):
        video_info, data_path = [], Path(data_dir)  # 결과저장용 리스트 초기화 Path를 통해 저장
        video_extensions = ["*.mp4", "*.avi", "*.mov"]  # 확장자 패턴
        for label, sub_dir_name in enumerate(["real", "fake"]):  # real: 0 , fake: 1
            sub_dir = data_path / sub_dir_name  # real, fake 디렉토리
            if sub_dir.exists():
                for ext in video_extensions:
                    for vf in sub_dir.glob(ext):  # 해당 패턴에 맞는 모든 파일 경로추적
                        video_info.append((str(vf), label, f"{sub_dir_name}_{vf.stem}"))  # vf.stem은 확장자 없는 파일형태
        if not video_info: raise ValueError(f"비디오 파일 없음: {data_dir}")
        return video_info

    def process_dataset(self, data_dir, output_dir, num_workers=4):
        # 중간 저장 단계 pickle 파일경로, 최종 저장 WebDataset 경로 설정
        output_path, temp_otp_path = Path(output_dir), Path(output_dir) / "temp_processed"
        # 임시 없으면 생성
        temp_otp_path.mkdir(exist_ok=True, parents=True)
        # real/fake 디렉토리 스캔
        video_info_list = self.create_dataset_metadata(data_dir)
        all_processed_data = []  # 전체 데이터 저장 리스트 초기화

        # 순차 처리로 변경 (데드락 방지)
        for info in tqdm(video_info_list, desc="1/2단계: 얼굴 검출 및 추출 중...."):
            try:
                res_list = self.process_video_and_ext_faces(info, temp_otp_path)
                if res_list:
                    all_processed_data.extend(res_list)
            except Exception as e:
                self.logger.error(f"비디오 결과 처리 오류 {e}")

        # for 루프가 끝난 후, 모든 비디오 처리가 완료된 시점에서 로그 출력 및 저장 함수 호출
        self.logger.info(f"총 {len(all_processed_data)}개의 얼굴 시퀀스 처리 , webDataset 생성 시작")
        self.save_as_webdataset(all_processed_data, output_path)
        return output_path

    def save_as_webdataset(self, processed_data, output_dir):
        # 데이터를 나누기에 샘플 수가 너무 적으면 함수를 즉시 종료
        if len(processed_data) < 2:
            self.logger.warning(f"데이터셋을 나누기에 샘플이 부족합니다. (총 {len(processed_data)}개). WebDataset 생성을 건너뜁니다.")
            return

        labels = [d["label"] for d in processed_data]  # 각 아이템에서 레이블만 추출
        # stratify: 학습, 검증 세트에 원본데이터 클래스 분포 유지하도록 샘플링
        use_stratify = len(labels) > 10 and len(set(labels)) > 1  # 샘플이 10개이상, 레이블 종류 2개이상이면 stratify
        try:
            train_data, temp_data, train_labels, temp_labels = train_test_split(
                processed_data, labels, test_size=0.2, random_state=42, stratify=labels if use_stratify else None
                # 학습 80% 임시셋: 20%
            )
            val_data, test_data = train_test_split(
                temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels if use_stratify else None
                # 임시셋 20%에서 10:10 학습/ 검증 데이터 분할
            )
        except ValueError:
            # 에러시 stratify없이 80:20 분리
            train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
            # temp_data를 val_data와 test_data로 분할
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        splits = {"train": train_data, "val": val_data, "test": test_data}
        # 각각의 세트에 대해 WebDatast 파일 생성
        for split_name, split_data in splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            # 경로의 백슬래시를 슬래시로 변경하고, 공백 제거
            raw_path = str(split_dir / f"{split_name}-%06d.tar")
            posix_path = raw_path.replace("\\", "/")
            output_shared_path = f"file:{posix_path}"  # Tar파일 패턴
            # 학습은 랜덤 증강, 검증/태스트 고정 증강
            transform_to_apply = self.train_transform if split_name == "train" else self.val_transform
            with wds.ShardWriter(output_shared_path, maxcount=1000) as writer:  # 최대 레코드 1000
                for item in tqdm(split_data, desc=f"2/2단계: '{split_name}' WebDataset 생성 중..."):
                    processed_seq = []  # 하나의 얼굴 시퀀스 저장할 임시 리스트
                    for face_img in item["faces"]:
                        try:
                            transformed = transform_to_apply(image=face_img)  # 이미지 증강 후 딕셔너리 형태로
                            processed_seq.append(transformed["image"])  # 전처리 된것만 리스트 추가
                        except Exception:
                            continue
                    if len(processed_seq) != self.config["seq_length"]:
                        continue
                    seq_tensor = torch.stack(processed_seq)  # 리스트형 텐서를 하나의 (bxCxHxW)형태로
                    sample = {
                        "__key__": item["video_id"],  # 고유 샘플 키
                        "seq.pth": seq_tensor,  # 얼굴시퀀스 텐서
                        "json": {  # 메타정보 데이터
                            "label": item["label"],
                            "original_video_path": item["original_video_path"]
                        }
                    }
                    writer.write(sample)  # ShardWriter에 샘플 기록 Tar 파일에 저장
            self.logger.info(f"'{split_name}' WebDataset 생성 완료. 경로: {split_dir}")


def create_deepfake_webdataset(tar_path: str, shuffle_size=1000, batch_size=16, num_workers=4):
    # 디렉토리 경로
    path = Path(tar_path)
    # 지정 확장자의 webdataset 파일 경로를 리스트화
    urls = [str(p) for p in path.glob("*.tar")]
    if not urls:
        raise FileNotFoundError(f"지정된 경로에 tar파일이 없습니다: {tar_path}")

    dataset = (
        # wds 객체생성(resampled=True: 각 epoch마다 랜덤 샘플링된 순서로 데이터 재배치)
        wds.WebDataset(urls, resampled=True)
        .shuffle(shuffle_size)  # 지정된 버퍼크기만큼 무작위로 섞음
        .decode(wds.torch_loads)  # 직렬화된 데이터를 디코딩
        .to_tuple("seq.pth", "json")  # 튜플 형태의 pth, json-> pth가 전처리 데이터, json-메타데이터
        .map_tuple(lambda x: x, lambda y: y["label"])  # json에서 label만 추출, (tensor,label) 형태로
    )
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,  # 지정 배치 사이즈
        num_workers=num_workers,  # 병렬 워커
        drop_last=False,  # 마지막 남은 샘플도 반호나
        shuffle=False,  # 추가 셔플 False
        persistent_workers=True  # 학습 속도 개선
    )
    return loader.unbatched().batched(batch_size)  # 균일한 배치


def main():
    config = {
        "image_size": 224,  # output 이미지 사이지 (224 x 224)
        "max_fps": 32,  # 최대 프레임
        "seq_length": 16,  # 시퀀스 길이
        "det_size": (320, 320),  # 검출 해상도
        "gpu_id": 0,  # gpu 사용
        'face_margin': 20,  # 얼굴 마진
        "num_workers":  8  # 병렬 워커 수
    }

    preprocessor = DeepFake_Data_Preprocessing(config)

    data_dir = "D:/PythonProject2/input_data"
    output_dir = "D:/PythonProject2/ouput_dir4"

    # data_dir 경로가 실제로 존재하는지 확인
    if not Path(data_dir).exists():
        print(f"오류: 입력 데이터 디렉토리가 유효하지 않습니다: {data_dir}")
        return

    processed_output_path = preprocessor.process_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        num_workers=config["num_workers"]
    )

    print("\n--- WebDataset 생성 완료 ---")
    print(f"전처리된 데이터가 다음 경로에 .tar 파일로 저장되었습니다: {processed_output_path}")

    print("\n--- 'train' Webdataset 로딩 테스트 코드 ---")

    train_path = Path(output_dir) / "train"
    if train_path.exists():
        try:
            train_loader = create_deepfake_webdataset(
                tar_path=str(train_path),
                batch_size=16,  # 훈련 시 배치 크기- 커스텀 가능
                num_workers=config["num_workers"]  # 워커 수 커스텀 가능
            )
            print("--- 'train' Webdataset 로딩 테스트 성공 ---")
        except (FileNotFoundError, StopIteration) as e:
            print(f"WebDataset 테스트 중 오류: {e}")
            print("생성된 데이터가 너무 적거나 .tar 파일 경로가 올바른지 확인하세요.")
        except Exception as e:
            print(f"알 수 없는 오류 발생: {e}")
    else:
        print("--- 'train' Webdataset이 생성되지 않아 로딩 테스트를 건너뜁니다. ---")


if __name__ == "__main__":
    main()