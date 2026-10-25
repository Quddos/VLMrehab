// RehabVLM Unity client.
//
// 1. Get a free API key at https://rehabvlm.vercel.app/developers.html
// 2. Add this component to any GameObject and assign a Humanoid Animator (an avatar driven
//    by VR body tracking, Kinect, a webcam pose solver, mocap, ...).
// 3. Call Assess("assess the shoulders", result => Debug.Log(result.answer));
//    or tick "Auto Assess" to poll every few seconds.
//
// Joint positions are sent in the avatar's root space (Y up, avatar facing +Z), so the result
// does not depend on where the player stands or which way the room is oriented.
using System;
using System.Collections;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace RehabVLM
{
    [Serializable]
    public class JointAngles
    {
        public float r_shoulder_abd, l_shoulder_abd;
        public float r_elbow_flex, l_elbow_flex;
        public float r_hip_abd, l_hip_abd;
        public float trunk_lean;
    }

    [Serializable]
    public class AssessResult
    {
        public string model;
        public string answer;          // natural-language answer from the VLM
        public JointAngles angles_deg; // direct geometric measurements, handy for gameplay
        public int latency_ms;
        [NonSerialized] public string rawJson;
        [NonSerialized] public string error;
    }

    public class RehabVLMClient : MonoBehaviour
    {
        [Tooltip("Your rvlm_... key. Keep it out of public repositories.")]
        public string apiKey = "";
        public string endpoint = "https://rehabvlm.vercel.app/api/v1/assess";
        public Animator humanoid;

        [Header("Optional automatic polling")]
        public bool autoAssess = false;
        public string autoQuestion = "assess the shoulders";
        [Min(1f)] public float intervalSeconds = 5f;
        public event Action<AssessResult> OnResult;

        static readonly (string name, HumanBodyBones bone)[] Map =
        {
            ("nose", HumanBodyBones.Head),
            ("l_shoulder", HumanBodyBones.LeftUpperArm), ("r_shoulder", HumanBodyBones.RightUpperArm),
            ("l_elbow", HumanBodyBones.LeftLowerArm), ("r_elbow", HumanBodyBones.RightLowerArm),
            ("l_wrist", HumanBodyBones.LeftHand), ("r_wrist", HumanBodyBones.RightHand),
            ("l_hip", HumanBodyBones.LeftUpperLeg), ("r_hip", HumanBodyBones.RightUpperLeg),
            ("l_knee", HumanBodyBones.LeftLowerLeg), ("r_knee", HumanBodyBones.RightLowerLeg),
            ("l_ankle", HumanBodyBones.LeftFoot), ("r_ankle", HumanBodyBones.RightFoot),
        };

        float _next;
        bool _inFlight;

        void Update()
        {
            if (!autoAssess || _inFlight || Time.time < _next) return;
            _next = Time.time + intervalSeconds;
            Assess(autoQuestion, r => OnResult?.Invoke(r));
        }

        /// <summary>Send the avatar's current pose with a question; the callback runs on the main thread.</summary>
        public void Assess(string question, Action<AssessResult> onDone)
        {
            StartCoroutine(AssessRoutine(question, onDone));
        }

        public string BuildRequestJson(string question)
        {
            var root = humanoid.transform;
            var sb = new StringBuilder("{\"space\":\"unity\",\"question\":\"");
            sb.Append(question.Replace("\\", "\\\\").Replace("\"", "\\\"")).Append("\",\"joints\":{");
            for (int i = 0; i < Map.Length; i++)
            {
                var t = humanoid.GetBoneTransform(Map[i].bone);
                if (t == null) throw new InvalidOperationException($"Avatar has no {Map[i].bone} bone");
                var p = root.InverseTransformPoint(t.position);
                if (Map[i].name == "nose") p += Vector3.up * 0.1f; // head bone sits at the skull base
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(Map[i].name).Append("\":[")
                  .Append(p.x.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
                  .Append(p.y.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
                  .Append(p.z.ToString("F4", CultureInfo.InvariantCulture)).Append(']');
            }
            return sb.Append("}}").ToString();
        }

        IEnumerator AssessRoutine(string question, Action<AssessResult> onDone)
        {
            _inFlight = true;
            var body = Encoding.UTF8.GetBytes(BuildRequestJson(question));
            using (var req = new UnityWebRequest(endpoint, "POST"))
            {
                req.uploadHandler = new UploadHandlerRaw(body);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                req.SetRequestHeader("Authorization", "Bearer " + apiKey);
                req.timeout = 30;
                yield return req.SendWebRequest();

                AssessResult result;
                if (req.result == UnityWebRequest.Result.Success)
                {
                    result = JsonUtility.FromJson<AssessResult>(req.downloadHandler.text);
                }
                else
                {
                    result = new AssessResult { error = $"{req.responseCode} {req.error}: {req.downloadHandler.text}" };
                }
                result.rawJson = req.downloadHandler.text;
                _inFlight = false;
                onDone?.Invoke(result);
            }
        }
    }
}
