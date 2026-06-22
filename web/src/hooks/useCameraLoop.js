import { useEffect, useRef, useState } from "react";

// カメラ映像の「映す→AIで処理する→表示する」をぐるぐる繰り返す共通の仕組み。
// Python の「while cap.isOpened(): ...」のループに相当します。
//
// 使い方:
//   const { videoRef, canvasRef, status } = useCameraLoop({
//     running,                 // true の間だけカメラを動かす
//     setup: async () => 専門家,// 最初に1回だけ呼ばれる（AIモデルの準備など）
//     process: (ctx, video, 専門家) => { ...毎フレームの処理... },
//   });
export function useCameraLoop({ running, setup, process }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("idle"); // idle / loading / running / error
  const [error, setError] = useState(null);

  // 毎回作り直される関数でも「最新版」を呼べるように ref に保管しておく
  const setupRef = useRef(setup);
  const processRef = useRef(process);
  setupRef.current = setup;
  processRef.current = process;

  useEffect(() => {
    if (!running) {
      setStatus("idle");
      return;
    }

    let stopped = false;
    let stream = null;
    let raf = 0;
    let detector = null;

    async function begin() {
      try {
        setStatus("loading");
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        // 1) AIモデルなどの準備（時間がかかることがある）
        detector = setupRef.current ? await setupRef.current() : null;
        if (stopped) return;

        // 2) PCのカメラを起動する（Python の cv2.VideoCapture(0) 相当）
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 1280, height: 720 },
          audio: false,
        });
        if (stopped) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        video.srcObject = stream;
        await video.play();
        setStatus("running");

        // 3) 1フレームずつ繰り返す
        const loop = () => {
          if (stopped) return;
          if (video.videoWidth > 0) {
            // キャンバスの大きさを映像に合わせる
            if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
            if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

            // 映像を鏡のように左右反転して描く（Python の cv2.flip(image, 1) 相当）
            ctx.save();
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            ctx.restore();

            // 各ステップ独自の処理（顔検出・手検出など）を呼ぶ
            try {
              processRef.current?.(ctx, video, detector);
            } catch (e) {
              console.error(e);
            }
          }
          raf = requestAnimationFrame(loop);
        };
        loop();
      } catch (e) {
        console.error(e);
        setError(e?.message || String(e));
        setStatus("error");
      }
    }

    begin();

    // running が false になったら、お片付け（Python の cap.release() 相当）
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      detector?.close?.();
    };
  }, [running]);

  return { videoRef, canvasRef, status, error };
}
