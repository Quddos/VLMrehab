// Example: show the VLM's feedback on a world-space TextMesh (or any UI) above the player.
using UnityEngine;

namespace RehabVLM
{
    public class RehabVLMExample : MonoBehaviour
    {
        public RehabVLMClient client;
        public TextMesh feedback;

        void Start()
        {
            client.OnResult += r =>
            {
                if (r.error != null) { Debug.LogWarning(r.error); return; }
                feedback.text = r.answer;
                // Gameplay hook: e.g. reward the player once the weaker arm passes 120 degrees.
                if (Mathf.Min(r.angles_deg.l_shoulder_abd, r.angles_deg.r_shoulder_abd) > 120f)
                    Debug.Log("Goal reached!");
            };
        }

        // Call from a button / gesture.
        public void AskForExercises() => client.Assess("suggest exercises", r => feedback.text = r.answer);
    }
}
