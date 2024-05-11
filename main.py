import cv2
import mediapipe 
from random import randint
from numpy import where, ndarray
import imutils
import streamlit as st 

def GenWall(h: int) -> int:
    return randint(101, h-40)

def DrawWallBuffer(img: ndarray, wallBuf: list) -> None:
    """Draw Wall Buffer To img Buffer"""
    h, w, c = img.shape
    for wall in wallBuf:
        cv2.rectangle(img, (wall[0], h), (wall[0]+40, wall[1]), (0, 255, 0), cv2.FILLED) # bottom wall 
        cv2.rectangle(img, (wall[0], 0), (wall[0]+40, wall[1]-100), (0, 255, 0), cv2.FILLED) # top wall

def MoveWall(wallBuf) -> list:
    """Move Wall Right to Left, 10px per frame"""
    newWallBuf = list()
    for wall in wallBuf:
        newWallBuf.append((wall[0]-10, wall[1]))
    return newWallBuf

def isCollision(x: int, y: int, wallBuf: list) -> bool:
    """Check of Collision bird to wall"""
    for wall in wallBuf:
        if x in range(wall[0], wall[0]+40): # bird in location of wall
            if not y in range(wall[1]-100, wall[1]): # bird not in space between top and bottom wall (Collision)
                return True
    return False

def ShowBird(img: ndarray, x: int, y: int, bird: ndarray) -> None:
    """show bird in pose nose of player"""
    y = y - 20
    x = x - 30
    # delete the 0 px for transparent
    con = bird != 0
    out = where(con, bird, img[y:y+50, x:x+50])
    img[y:y+50, x:x+50] = out[:]

def ReadBirdFile() -> ndarray:
    """read and resize bird photo"""
    bird = cv2.imread("gallery/sprites/bird.png")
    bird = cv2.resize(bird, [50,50])
    return bird

def GameOver(score: int, img: ndarray) -> None:
    """Show `Game Over!` message and `player score` then exit the game"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay = img.copy()
    alpha = 0.6  # Transparency factor
    cv2.rectangle(overlay, (50, 150), (550, 350), (0, 0, 0), cv2.FILLED)  # Background rectangle
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, "Game Over!", (150, 220), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Final Score: {score}", (150, 270), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Press any key to exit", (150, 320), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Flappy Bird Game!", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)

def main() -> None:
    """Setup"""
    face = mediapipe.solutions.face_detection.FaceDetection()
    cap = cv2.VideoCapture(0)
    bird = ReadBirdFile()
    frame = 0
    score = 0
    wallBuf = list()
    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success: 
                print("[ERROR] Read Capture Failed.")
                break
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face.process(imgRGB)
            if not res.detections: GameOver(score, img) # Face Not Found in image, To prevent fraud 
            nose = res.detections[0].location_data.relative_keypoints[2] # get position of player nose
            nose.x, nose.y = nose.x * w, nose.y * h # Convert float position to Real position in Image

            ShowBird(img, int(nose.x), int(nose.y), bird)

            frame += 1
            if frame >= 30: # Generate a New Wall at 30 frame
                frame = 0
                wallBuf.append((w-50, GenWall(h)))

            DrawWallBuffer(img, wallBuf)

            wallBuf = MoveWall(wallBuf)
            
            if isCollision(int(nose.x), int(nose.y), wallBuf):
                GameOver(score, img)
            else:
                score += 1

            cv2.putText(img, str(score), (5,20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))

            cv2.imshow("Flappy Bird Game!", img)
            cv2.waitKey(30)

    except KeyboardInterrupt:
        GameOver(score, img)


st.set_page_config(page_title="Flap-to-Fit",
                   page_icon="gallery/sprites/bird.png",
                   )


st.title(" :rainbow[Flap-to-Fit: Nose-Controlled Game]")
st.write("Stay active while having fun with our Flap-to-Fit game!")


st.image("gallery/sprites/game_image.png", use_column_width=True)


st.write("Use your nose to control the game! Move up and down to navigate through obstacles.")


if st.button("Start Game", type="primary"):
    st.write("Game is starting...")
    main()


