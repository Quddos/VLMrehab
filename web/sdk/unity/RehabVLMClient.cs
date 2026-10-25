// RehabVLM Unity client.
//
// 1. Get a free API key at https://rehabvlm.vercel.app/developers.html and provide it in ONE of:
//      - the REHABVLM_API_KEY environment variable (editor, desktop and server builds)
//      - a git-ignored Assets/Resources/rehabvlm_env.txt containing REHABVLM_API_KEY=rvlm_...
//        (works on Quest/Android/iOS too)
//      - the Api Key field in the Inspector
// 2. Add this component to any GameObject and assign a Humanoid Animator (an avatar driven
//    by VR body tracking, Kinect, a webcam pose solver, mocap, ...).
// 3. Call Assess("assess the shoulders", result => Debug.Log(result.answer));
//    or tick "Auto Assess" to poll every few seconds.
// For hand tracking use RehabVLMHandClient (needs the com.unity.xr.hands package).
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
    /// <summary>Finds the API key: Inspector value, then REHABVLM_API_KEY env var, then Resources/rehabvlm_env.txt.</summary>
    public static class RehabVLMConfig
    {
        public const string EnvVar = "REHABVLM_API_KEY";
        public const string DefaultBaseUrl = "https://rehabvlm.vercel.app";

        public static string ResolveKey(string inspectorValue)
        {
            if (!string.IsNullOrWhiteSpace(inspectorValue)) return inspectorValue.Trim();
            var env = Environment.GetEnvironmentVariable(EnvVar);
            if (!string.IsNullOrWhiteSpace(env)) return env.Trim();
            var file = Resources.Load<TextAsset>("rehabvlm_env");
            if (file != null)
            {
                foreach (var line in file.text.Split('\n'))
                {
                    var t = line.Trim();
                    if (t.StartsWith(EnvVar + "=")) return t.Substring(EnvVar.Length + 1).Trim().Trim('"');
                }
            }
            return null;
        }

        public static string Num(float v) => v.ToString("F4", CultureInfo.InvariantCulture);

        /// <summary>POST JSON to the API; the callback receives (statusCode, body, error).</summary>
        public static IEnumerator Post(string url, string apiKey, string json, Action<long, string, string> done)
        {
            var key = ResolveKey(apiKey);
            if (key == null)
            {
                done(0, null, $"No API key: set {EnvVar}, add Resources/rehabvlm_env.txt or fill the Api Key field.");
                yield break;
            }
            using (var req = new UnityWebRequest(url, "POST"))
            {
                req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                req.SetRequestHeader("Authorization", "Bearer " + key);
                req.timeout = 30;
                yield return req.SendWebRequest();
                var ok = req.result == UnityWebRequest.Result.Success;
                done(req.responseCode, req.downloadHandler.text, ok ? null : req.error);
            }
        }

        public static string Escape(string s) => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

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
        [Tooltip("Optional. Leave empty to use the REHABVLM_API_KEY env var or Resources/rehabvlm_env.txt.")]
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
            sb.Append(RehabVLMConfig.Escape(question)).Append("\",\"joints\":{");
            for (int i = 0; i < Map.Length; i++)
            {
                var t = humanoid.GetBoneTransform(Map[i].bone);
                if (t == null) throw new InvalidOperationException($"Avatar has no {Map[i].bone} bone");
                var p = root.InverseTransformPoint(t.position);
                if (Map[i].name == "nose") p += Vector3.up * 0.1f; // head bone sits at the skull base
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(Map[i].name).Append("\":[")
                  .Append(RehabVLMConfig.Num(p.x)).Append(',')
                  .Append(RehabVLMConfig.Num(p.y)).Append(',')
                  .Append(RehabVLMConfig.Num(p.z)).Append(']');
            }
            return sb.Append("}}").ToString();
        }

        IEnumerator AssessRoutine(string question, Action<AssessResult> onDone)
        {
            _inFlight = true;
            yield return RehabVLMConfig.Post(endpoint, apiKey, BuildRequestJson(question), (code, text, err) =>
            {
                var result = err == null ? JsonUtility.FromJson<AssessResult>(text)
                                         : new AssessResult { error = $"{code} {err}: {text}" };
                result.rawJson = text;
                _inFlight = false;
                onDone?.Invoke(result);
            });
        }
    }
}
