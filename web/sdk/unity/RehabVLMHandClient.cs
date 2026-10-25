// RehabVLM hand client for Unity XR hand tracking (Meta Quest, Pico, Vision Pro, Ultraleap, ...).
//
// Requires the XR Hands package (com.unity.xr.hands) and RehabVLMClient.cs (for RehabVLMConfig).
// The API key is resolved the same way as RehabVLMClient: Inspector field, REHABVLM_API_KEY env var,
// or a git-ignored Assets/Resources/rehabvlm_env.txt containing REHABVLM_API_KEY=rvlm_...
//
//   handClient.Assess("make a fist", r => label.text = r.answer);
//
// Not using XR Hands? Call AssessLandmarks() with 21 points in MediaPipe Hands order instead
// (e.g. from a webcam hand tracker).
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace RehabVLM
{
    [Serializable]
    public class HandMeasurements
    {
        public float thumb_flex, index_flex, middle_flex, ring_flex, pinky_flex; // degrees, summed over finger joints
        public float spread;     // degrees between index and pinky; 0 when fingers are too curled to measure
        public float pinch_gap;  // thumb tip to index tip, % of hand width
    }

    [Serializable]
    public class HandAssessResult
    {
        public string model;
        public string answer;
        public HandMeasurements measurements;
        public string handedness;
        public int latency_ms;
        [NonSerialized] public string rawJson;
        [NonSerialized] public string error;
    }

    public class RehabVLMHandClient : MonoBehaviour
    {
        [Tooltip("Optional. Leave empty to use the REHABVLM_API_KEY env var or Resources/rehabvlm_env.txt.")]
        public string apiKey = "";
        public string endpoint = RehabVLMConfig.DefaultBaseUrl + "/api/v1/assess";
        public Handedness hand = Handedness.Right;

        [Header("Optional automatic polling")]
        public bool autoAssess = false;
        public string autoQuestion = "assess the fist";
        [Min(1f)] public float intervalSeconds = 3f;
        public event Action<HandAssessResult> OnResult;

        // XR Hands joints in MediaPipe Hands order (wrist, thumb CMC-MCP-IP-tip, then MCP-PIP-DIP-tip per finger)
        static readonly XRHandJointID[] Order =
        {
            XRHandJointID.Wrist,
            XRHandJointID.ThumbMetacarpal, XRHandJointID.ThumbProximal, XRHandJointID.ThumbDistal, XRHandJointID.ThumbTip,
            XRHandJointID.IndexProximal, XRHandJointID.IndexIntermediate, XRHandJointID.IndexDistal, XRHandJointID.IndexTip,
            XRHandJointID.MiddleProximal, XRHandJointID.MiddleIntermediate, XRHandJointID.MiddleDistal, XRHandJointID.MiddleTip,
            XRHandJointID.RingProximal, XRHandJointID.RingIntermediate, XRHandJointID.RingDistal, XRHandJointID.RingTip,
            XRHandJointID.LittleProximal, XRHandJointID.LittleIntermediate, XRHandJointID.LittleDistal, XRHandJointID.LittleTip,
        };

        XRHandSubsystem _subsystem;
        float _next;
        bool _inFlight;

        void Update()
        {
            if (!autoAssess || _inFlight || Time.time < _next) return;
            _next = Time.time + intervalSeconds;
            Assess(autoQuestion, r => OnResult?.Invoke(r));
        }

        /// <summary>Read the tracked hand from XR Hands and ask a question about it.</summary>
        public void Assess(string question, Action<HandAssessResult> onDone)
        {
            var points = ReadTrackedHand();
            if (points == null)
            {
                onDone?.Invoke(new HandAssessResult { error = $"{hand} hand is not tracked" });
                return;
            }
            AssessLandmarks(points, question, onDone, hand == Handedness.Left ? "left" : "right");
        }

        /// <summary>Ask about any 21 hand landmarks in MediaPipe Hands order (Unity world or local space).</summary>
        public void AssessLandmarks(IList<Vector3> points, string question, Action<HandAssessResult> onDone,
                                    string handedness = "right")
        {
            if (points == null || points.Count != 21) throw new ArgumentException("Need exactly 21 hand landmarks");
            var sb = new StringBuilder("{\"space\":\"unity\",\"handedness\":\"").Append(handedness)
                .Append("\",\"question\":\"").Append(RehabVLMConfig.Escape(question)).Append("\",\"hand\":[");
            for (int i = 0; i < 21; i++)
            {
                var p = points[i];
                if (i > 0) sb.Append(',');
                sb.Append('[').Append(RehabVLMConfig.Num(p.x)).Append(',').Append(RehabVLMConfig.Num(p.y))
                  .Append(',').Append(RehabVLMConfig.Num(p.z)).Append(']');
            }
            StartCoroutine(Send(sb.Append("]}").ToString(), onDone));
        }

        Vector3[] ReadTrackedHand()
        {
            if (_subsystem == null || !_subsystem.running)
            {
                var all = new List<XRHandSubsystem>();
                SubsystemManager.GetSubsystems(all);
                _subsystem = all.Find(s => s.running);
                if (_subsystem == null) return null;
            }
            var xrHand = hand == Handedness.Left ? _subsystem.leftHand : _subsystem.rightHand;
            if (!xrHand.isTracked) return null;
            var points = new Vector3[21];
            for (int i = 0; i < Order.Length; i++)
            {
                if (!xrHand.GetJoint(Order[i]).TryGetPose(out var pose)) return null;
                points[i] = pose.position;
            }
            return points;
        }

        IEnumerator Send(string json, Action<HandAssessResult> onDone)
        {
            _inFlight = true;
            yield return RehabVLMConfig.Post(endpoint, apiKey, json, (code, text, err) =>
            {
                var result = err == null ? JsonUtility.FromJson<HandAssessResult>(text)
                                         : new HandAssessResult { error = $"{code} {err}: {text}" };
                result.rawJson = text;
                _inFlight = false;
                onDone?.Invoke(result);
            });
        }
    }
}
