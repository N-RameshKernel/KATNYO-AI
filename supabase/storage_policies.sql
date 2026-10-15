-- Create a private bucket named katnyo-user-data in Supabase Storage first.
-- Objects must be stored under: <auth.uid()>/<dataset>/<revision>/<filename>

drop policy if exists "katnyo users read own csv objects" on storage.objects;
create policy "katnyo users read own csv objects"
on storage.objects for select to authenticated
using (
  bucket_id = 'katnyo-user-data'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);

drop policy if exists "katnyo users upload own csv objects" on storage.objects;
create policy "katnyo users upload own csv objects"
on storage.objects for insert to authenticated
with check (
  bucket_id = 'katnyo-user-data'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);

drop policy if exists "katnyo users update own csv objects" on storage.objects;
create policy "katnyo users update own csv objects"
on storage.objects for update to authenticated
using (
  bucket_id = 'katnyo-user-data'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
)
with check (
  bucket_id = 'katnyo-user-data'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);

drop policy if exists "katnyo users delete own csv objects" on storage.objects;
create policy "katnyo users delete own csv objects"
on storage.objects for delete to authenticated
using (
  bucket_id = 'katnyo-user-data'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);
